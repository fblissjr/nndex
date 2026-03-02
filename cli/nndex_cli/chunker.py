"""File discovery and chunking for code search indexing.

Supports AST-aware chunking for Python (via ast module) and heuristic
definition-boundary chunking for JS/TS/Rust. Falls back to sliding
window for other file types or on parse failure.
"""

import ast
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional


@dataclass
class Chunk:
    """A chunk of text from a file with location metadata."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    preview: str


# Extensions that are almost certainly binary
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".wasm",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".sqlite", ".db", ".parquet", ".arrow",
    ".safetensors", ".bin", ".gguf", ".onnx", ".pt", ".pth",
})

# Directories always skipped
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".tox",
    ".venv", "venv", ".env",
    ".nndex",
})


def _parse_gitignore(root: Path) -> List[str]:
    """Parse .gitignore and return list of patterns."""
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for line in gitignore.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _is_gitignored(path: Path, root: Path, patterns: List[str]) -> bool:
    """Check if a path matches any gitignore pattern."""
    rel = str(path.relative_to(root))
    name = path.name
    for pattern in patterns:
        # Directory pattern (trailing slash)
        clean = pattern.rstrip("/")
        if fnmatch(name, clean) or fnmatch(rel, clean) or fnmatch(rel, clean + "/*"):
            return True
        if pattern.endswith("/") and path.is_dir() and fnmatch(name, clean):
            return True
    return False


def _is_binary(path: Path) -> bool:
    """Heuristic check if a file is binary."""
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
    except (OSError, PermissionError):
        return True
    return False


def discover_files(root: Path) -> List[Path]:
    """Recursively find text files, respecting .gitignore and skipping binaries.

    Args:
        root: Directory to search from.

    Returns:
        Sorted list of text file paths.
    """
    root = root.resolve()
    patterns = _parse_gitignore(root)
    results = []

    def _walk(directory: Path):
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            return

        for entry in entries:
            # Skip hidden files and directories
            if entry.name.startswith("."):
                continue

            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                if _is_gitignored(entry, root, patterns):
                    continue
                _walk(entry)
            elif entry.is_file():
                if _is_gitignored(entry, root, patterns):
                    continue
                if _is_binary(entry):
                    continue
                results.append(entry)

    _walk(root)
    return results


def _python_definition_lines(source: str) -> List[int]:
    """Use Python AST to find top-level definition start lines.

    Returns sorted list of 1-based line numbers where top-level
    functions, classes, or async functions start.
    Returns empty list on syntax error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines.append(node.lineno)
    return sorted(lines)


# Heuristic patterns for definition boundaries in non-Python languages.
# These match the start of top-level definitions (not indented).
_DEFINITION_PATTERNS = {
    # Rust: fn, struct, enum, impl, mod, trait, type, const, static
    ".rs": re.compile(
        r'^(?:pub(?:\(crate\))?\s+)?(?:async\s+)?(?:unsafe\s+)?'
        r'(?:fn|struct|enum|impl|mod|trait|type|const|static)\s',
    ),
    # JS/TS: function, class, const/let/var (arrow fns), export variants
    ".js": re.compile(
        r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class)\s',
    ),
    ".ts": re.compile(
        r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum)\s',
    ),
    ".jsx": re.compile(
        r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class)\s',
    ),
    ".tsx": re.compile(
        r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum)\s',
    ),
}


def _heuristic_definition_lines(source: str, suffix: str) -> List[int]:
    """Find definition boundary lines using regex heuristics.

    Returns sorted list of 1-based line numbers.
    """
    pattern = _DEFINITION_PATTERNS.get(suffix)
    if pattern is None:
        return []

    lines = []
    for i, line in enumerate(source.split("\n"), 1):
        if pattern.match(line):
            lines.append(i)
    return lines


def _get_definition_boundaries(source: str, suffix: str) -> List[int]:
    """Get definition boundary lines for a file.

    Uses AST for Python, heuristics for JS/TS/Rust.
    Returns empty list if no boundaries found (falls back to sliding window).
    """
    if suffix == ".py":
        return _python_definition_lines(source)
    return _heuristic_definition_lines(source, suffix)


def _chunk_by_definitions(
    lines: List[str],
    file_path: str,
    boundaries: List[int],
    max_chars: int,
) -> List[Chunk]:
    """Split file into chunks aligned to definition boundaries.

    Groups consecutive definitions until adding the next one would
    exceed max_chars, then starts a new chunk. No overlap between chunks.
    """
    if not boundaries:
        return []

    # Convert 1-based boundary lines to 0-based indices into lines list
    # Each segment: [boundary_start, next_boundary_start) in the lines list
    segments = []
    for i, boundary in enumerate(boundaries):
        start = boundary - 1  # 0-based
        if i + 1 < len(boundaries):
            end = boundaries[i + 1] - 1
        else:
            end = len(lines)
        segments.append((start, end))

    # Preamble: lines before first definition (imports, comments, etc.)
    preamble_end = boundaries[0] - 1
    preamble = "\n".join(lines[:preamble_end]).rstrip()

    chunks = []
    current_lines_start = 0  # 0-based start of current chunk
    current_parts = []
    current_chars = 0

    if preamble:
        current_parts.append(preamble)
        current_chars = len(preamble)
        current_lines_start = 0

    for seg_start, seg_end in segments:
        segment_text = "\n".join(lines[seg_start:seg_end]).rstrip()
        segment_chars = len(segment_text)

        # If adding this segment would exceed max_chars and we already have content
        if current_parts and current_chars + segment_chars + 1 > max_chars:
            # Flush current chunk
            content = "\n".join(current_parts)
            # Recalculate actual line range from content
            chunk_line_count = content.count("\n") + 1
            preview = content[:100].replace("\n", " ").strip()
            chunks.append(Chunk(
                file_path=file_path,
                start_line=current_lines_start + 1,
                end_line=current_lines_start + chunk_line_count,
                content=content,
                preview=preview,
            ))
            # Start new chunk
            current_parts = [segment_text]
            current_chars = segment_chars
            current_lines_start = seg_start
        else:
            if not current_parts:
                current_lines_start = seg_start if not preamble else 0
            current_parts.append(segment_text)
            current_chars += segment_chars + 1

    # Flush remaining
    if current_parts:
        content = "\n".join(current_parts)
        chunk_line_count = content.count("\n") + 1
        preview = content[:100].replace("\n", " ").strip()
        chunks.append(Chunk(
            file_path=file_path,
            start_line=current_lines_start + 1,
            end_line=current_lines_start + chunk_line_count,
            content=content,
            preview=preview,
        ))

    return chunks


def chunk_file(
    path: Path,
    max_chars: int = 4000,
    overlap_lines: int = 5,
) -> List[Chunk]:
    """Split a file into chunks for embedding.

    Tries AST-aware chunking first (Python via ast, JS/TS/Rust via heuristics)
    to align chunk boundaries with code definitions. Falls back to a line-based
    sliding window for unsupported languages or when AST parsing fails.

    Args:
        path: File to chunk.
        max_chars: Approximate max characters per chunk.
        overlap_lines: Number of lines to overlap between chunks (sliding window only).

    Returns:
        List of Chunk objects.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return []

    if not text.strip():
        return []

    lines = text.split("\n")
    file_path = str(path)

    # Small file: single chunk
    if len(text) <= max_chars:
        preview = text[:100].replace("\n", " ").strip()
        return [Chunk(
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            content=text,
            preview=preview,
        )]

    # Try AST-aware chunking for supported languages
    suffix = path.suffix.lower()
    boundaries = _get_definition_boundaries(text, suffix)
    if boundaries:
        ast_chunks = _chunk_by_definitions(lines, file_path, boundaries, max_chars)
        if ast_chunks:
            return ast_chunks

    # Fallback: split into overlapping chunks by line count
    chunks = []
    i = 0
    while i < len(lines):
        # Collect lines until we hit max_chars
        chunk_lines = []
        char_count = 0
        j = i
        while j < len(lines) and char_count < max_chars:
            chunk_lines.append(lines[j])
            char_count += len(lines[j]) + 1  # +1 for newline
            j += 1

        content = "\n".join(chunk_lines)
        preview = content[:100].replace("\n", " ").strip()
        chunks.append(Chunk(
            file_path=file_path,
            start_line=i + 1,
            end_line=i + len(chunk_lines),
            content=content,
            preview=preview,
        ))

        # Advance by chunk size minus overlap
        advance = max(1, len(chunk_lines) - overlap_lines)
        i += advance

        # Stop if we've covered everything
        if j >= len(lines):
            break

    return chunks


def chunk_directory(root: Path, max_chars: int = 4000) -> List[Chunk]:
    """Discover and chunk all text files in a directory.

    Args:
        root: Directory to process.
        max_chars: Max characters per chunk.

    Returns:
        List of all chunks across all discovered files.
    """
    files = discover_files(root)
    chunks = []
    for f in files:
        chunks.extend(chunk_file(f, max_chars=max_chars))
    return chunks
