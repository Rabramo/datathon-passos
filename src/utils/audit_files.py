from __future__ import annotations

from pathlib import Path
from typing import Iterable


IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
}


def iter_project_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        yield path


def find_empty_files(root: Path) -> list[Path]:
    return sorted([path for path in iter_project_files(root) if path.stat().st_size == 0])


def find_whitespace_only_text_files(root: Path, extensions: tuple[str, ...] = (".py", ".md", ".txt", ".yaml", ".yml")) -> list[Path]:
    results: list[Path] = []

    for path in iter_project_files(root):
        if path.suffix.lower() not in extensions:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if content.strip() == "":
            results.append(path)

    return sorted(results)


if __name__ == "__main__":
    project_root = Path(".").resolve()

    empty_files = find_empty_files(project_root)
    blank_text_files = find_whitespace_only_text_files(project_root)

    print("=== Empty files (0 bytes) ===")
    for path in empty_files:
        print(path.relative_to(project_root))

    print("\n=== Text files with only whitespace ===")
    for path in blank_text_files:
        print(path.relative_to(project_root))