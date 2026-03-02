"""File discovery and chunking for code search indexing."""

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


def chunk_file(
    path: Path,
    max_chars: int = 4000,
    overlap_lines: int = 5,
) -> List[Chunk]:
    """Split a file into chunks for embedding.

    Uses a simple line-based sliding window. For small files, returns a single chunk.

    Args:
        path: File to chunk.
        max_chars: Approximate max characters per chunk.
        overlap_lines: Number of lines to overlap between chunks.

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

    # Split into overlapping chunks
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
