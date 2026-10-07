"""File system and path manipulation utilities."""

from pathlib import Path
import re


def ensure_directory_exists(directory_path: Path) -> Path:
    """Ensure that the given directory path exists, creating parents if necessary."""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_safe_filename(original_name: str) -> str:
    """Sanitize filename to prevent directory traversal or unsafe characters."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\.-]", "_", original_name)
    return sanitized
