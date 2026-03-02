"""Tests for file chunking logic."""

import os
import pytest
from pathlib import Path


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temp directory with sample files for testing."""
    # Python file
    (tmp_path / "auth.py").write_text(
        'def login(user, password):\n'
        '    """Authenticate a user."""\n'
        '    if not user:\n'
        '        raise ValueError("no user")\n'
        '    return check_credentials(user, password)\n'
        '\n'
        '\n'
        'def logout(session):\n'
        '    """End a session."""\n'
        '    session.invalidate()\n'
        '    return True\n'
    )

    # Markdown file
    (tmp_path / "README.md").write_text(
        '# My Project\n'
        '\n'
        'A sample project for testing.\n'
        '\n'
        '## Installation\n'
        '\n'
        'Run `pip install myproject`.\n'
    )

    # Binary file (should be skipped)
    (tmp_path / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

    # Nested file
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "utils.py").write_text(
        'def helper():\n'
        '    return 42\n'
    )

    # .gitignore
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")

    # File that should be ignored
    (tmp_path / "debug.log").write_text("some log data\n")

    # Directory that should be ignored
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "output.js").write_text("compiled output")

    return tmp_path


class TestDiscoverFiles:
    """Test file discovery with gitignore and binary filtering."""

    def test_finds_text_files(self, tmp_repo):
        from nndex_cli.chunker import discover_files

        files = discover_files(tmp_repo)
        names = {f.name for f in files}
        assert "auth.py" in names
        assert "README.md" in names
        assert "utils.py" in names

    def test_skips_binary_files(self, tmp_repo):
        from nndex_cli.chunker import discover_files

        files = discover_files(tmp_repo)
        names = {f.name for f in files}
        assert "image.png" not in names

    def test_respects_gitignore(self, tmp_repo):
        from nndex_cli.chunker import discover_files

        files = discover_files(tmp_repo)
        names = {f.name for f in files}
        assert "debug.log" not in names
        assert "output.js" not in names

    def test_skips_dotfiles_and_hidden_dirs(self, tmp_repo):
        from nndex_cli.chunker import discover_files

        hidden = tmp_repo / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("secret = 42\n")

        files = discover_files(tmp_repo)
        names = {f.name for f in files}
        assert "secret.py" not in names


class TestChunkFile:
    """Test splitting files into embeddable chunks."""

    def test_small_file_single_chunk(self, tmp_repo):
        from nndex_cli.chunker import chunk_file

        chunks = chunk_file(tmp_repo / "src" / "utils.py")
        assert len(chunks) == 1
        assert chunks[0].file_path == str(tmp_repo / "src" / "utils.py")
        assert chunks[0].start_line == 1
        assert "helper" in chunks[0].content

    def test_chunk_has_metadata(self, tmp_repo):
        from nndex_cli.chunker import chunk_file

        chunks = chunk_file(tmp_repo / "auth.py")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.file_path
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.content
            assert chunk.preview  # Short preview text

    def test_large_file_multiple_chunks(self, tmp_path):
        from nndex_cli.chunker import chunk_file

        # Create a large file that exceeds chunk size
        lines = [f"def func_{i}():\n    return {i}\n\n" for i in range(200)]
        large_file = tmp_path / "large.py"
        large_file.write_text("".join(lines))

        chunks = chunk_file(large_file, max_chars=500)
        assert len(chunks) > 1

        # Verify no content is lost (overlap allowed)
        all_content = "".join(c.content for c in chunks)
        original = large_file.read_text()
        # Every line from original should appear in at least one chunk
        for line in original.strip().split("\n"):
            found = any(line in c.content for c in chunks)
            assert found, f"Line missing from chunks: {line[:50]}"


class TestASTChunking:
    """Test AST-aware chunking for Python/JS/Rust files.

    These tests verify that chunk boundaries align with code definition
    boundaries rather than splitting arbitrarily at character counts.
    """

    def test_python_function_not_split_across_chunks(self, tmp_path):
        """A function that fits in max_chars should never be split across chunks.

        The sliding window approach would split this in the middle of a function.
        AST-aware chunking should keep each function intact.
        """
        from nndex_cli.chunker import chunk_file

        py_file = tmp_path / "funcs.py"
        # Each function is ~80 chars. With max_chars=200, sliding window
        # would put the cut wherever 200 chars lands (possibly mid-function).
        # AST chunking should split between functions instead.
        py_file.write_text(
            'def alpha():\n'
            '    """Alpha function."""\n'
            '    x = 1\n'
            '    y = 2\n'
            '    return x + y\n'
            '\n'
            '\n'
            'def beta():\n'
            '    """Beta function."""\n'
            '    a = 10\n'
            '    b = 20\n'
            '    return a * b\n'
            '\n'
            '\n'
            'def gamma():\n'
            '    """Gamma function."""\n'
            '    p = 100\n'
            '    q = 200\n'
            '    return p - q\n'
        )
        chunks = chunk_file(py_file, max_chars=200)
        assert len(chunks) >= 2
        # Key assertion: no function definition should be split across chunks.
        # Each "def X():" and its "return" should be in the same chunk.
        for func, ret_val in [("alpha", "x + y"), ("beta", "a * b"), ("gamma", "p - q")]:
            chunks_with_def = [c for c in chunks if f"def {func}" in c.content]
            assert len(chunks_with_def) == 1, f"def {func} in {len(chunks_with_def)} chunks"
            assert ret_val in chunks_with_def[0].content, (
                f"{func}'s return value split from its definition"
            )

    def test_python_keeps_class_together(self, tmp_path):
        """A small class should stay in a single chunk."""
        from nndex_cli.chunker import chunk_file

        py_file = tmp_path / "myclass.py"
        py_file.write_text(
            'class MyClass:\n'
            '    def __init__(self):\n'
            '        self.value = 0\n'
            '\n'
            '    def increment(self):\n'
            '        self.value += 1\n'
        )
        chunks = chunk_file(py_file, max_chars=4000)
        assert len(chunks) == 1
        assert "MyClass" in chunks[0].content
        assert "increment" in chunks[0].content

    def test_python_falls_back_on_syntax_error(self, tmp_path):
        """Invalid Python should fall back to sliding window chunking."""
        from nndex_cli.chunker import chunk_file

        py_file = tmp_path / "bad.py"
        py_file.write_text(
            'def broken(\n'
            '    # missing closing paren and colon\n'
            '    return 1\n'
            '\n'
            'def also_broken(\n'
            '    return 2\n'
        )
        # Should not crash, falls back to line-based chunking
        chunks = chunk_file(py_file, max_chars=4000)
        assert len(chunks) >= 1
        assert "broken" in chunks[0].content

    def test_rust_fn_not_split_across_chunks(self, tmp_path):
        """Rust functions should not be split across chunks."""
        from nndex_cli.chunker import chunk_file

        rs_file = tmp_path / "lib.rs"
        rs_file.write_text(
            'use std::io;\n'
            '\n'
            'pub fn search(query: &str) -> Vec<usize> {\n'
            '    let mut results = Vec::new();\n'
            '    results\n'
            '}\n'
            '\n'
            'pub fn index(data: &[f32]) -> Index {\n'
            '    Index::new(data)\n'
            '}\n'
            '\n'
            'fn helper() -> bool {\n'
            '    true\n'
            '}\n'
        )
        chunks = chunk_file(rs_file, max_chars=150)
        assert len(chunks) >= 2
        # Each fn definition and its body should be in exactly one chunk
        for fn_name in ("search", "index", "helper"):
            chunks_with_fn = [c for c in chunks if f"fn {fn_name}" in c.content]
            assert len(chunks_with_fn) == 1, f"fn {fn_name} in {len(chunks_with_fn)} chunks"

    def test_js_function_not_split_across_chunks(self, tmp_path):
        """JS functions/classes should not be split across chunks."""
        from nndex_cli.chunker import chunk_file

        js_file = tmp_path / "app.js"
        js_file.write_text(
            'import { foo } from "bar";\n'
            '\n'
            'function handleRequest(req) {\n'
            '    return req.body;\n'
            '}\n'
            '\n'
            'class Router {\n'
            '    constructor() {\n'
            '        this.routes = [];\n'
            '    }\n'
            '}\n'
            '\n'
            'export function createApp() {\n'
            '    return new Router();\n'
            '}\n'
        )
        chunks = chunk_file(js_file, max_chars=150)
        assert len(chunks) >= 2
        # Each definition should be in exactly one chunk
        # (Use definition patterns to avoid matching references like "new Router()")
        for pattern in ("function handleRequest", "class Router", "function createApp"):
            matching = [c for c in chunks if pattern in c.content]
            assert len(matching) == 1, f"'{pattern}' in {len(matching)} chunks"


class TestChunkDirectory:
    """Test chunking all files in a directory."""

    def test_chunks_all_discovered_files(self, tmp_repo):
        from nndex_cli.chunker import chunk_directory

        chunks = chunk_directory(tmp_repo)
        # Should have chunks from auth.py, README.md, utils.py
        file_paths = {c.file_path for c in chunks}
        assert any("auth.py" in p for p in file_paths)
        assert any("README.md" in p for p in file_paths)
        assert any("utils.py" in p for p in file_paths)
        # Should NOT have chunks from binary or ignored files
        assert not any("image.png" in p for p in file_paths)
        assert not any("debug.log" in p for p in file_paths)
