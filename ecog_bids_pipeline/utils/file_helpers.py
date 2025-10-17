"""
Utility functions for file and directory operations.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

def setup_temp_dir(base_path: Optional[Union[str, Path]] = None, prefix: str = "bidsconv_") -> Path:
    """
    Create a temporary directory for intermediate processing.

    Args:
        base_path: Base directory for the temporary directory. If None, uses system temp dir.
        prefix: Prefix for the temporary directory name.

    Returns:
        Path to the created temporary directory.
    """
    base_path = Path(base_path) if base_path else Path(tempfile.gettempdir())
    base_path.mkdir(parents=True, exist_ok=True)

    # Create a uniquely named temporary directory
    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=base_path)
    return Path(temp_dir)

def cleanup_temp_dir(temp_dir: Union[str, Path]) -> None:
    """
    Safely remove a temporary directory and its contents.

    Args:
        temp_dir: Path to the temporary directory to remove.
    """
    temp_dir = Path(temp_dir)
    if temp_dir.exists() and temp_dir.is_dir():
        shutil.rmtree(temp_dir)

def copy_files_to_temp(
    source_file_map: Dict[str, Path],
    temp_dir: Union[str, Path]
) -> Dict[str, Path]:
    """
    Copy source files to a temporary directory.

    Args:
        source_file_map: Dictionary mapping file types to source file paths.
        temp_dir: Path to the temporary directory.

    Returns:
        Dictionary mapping the same keys to the new file paths in the temp directory.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_paths = {}
    for file_type, source_path in source_file_map.items():
        if source_path and source_path.exists():
            dest_path = temp_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            temp_paths[file_type] = dest_path
        else:
            temp_paths[file_type] = None

    return temp_paths

def find_file(
    directory: Union[str, Path],
    patterns: Union[str, list[str]],
    recursive: bool = False
) -> Optional[Path]:
    """
    Find the first file matching any of the given patterns.

    Args:
        directory: Directory to search in.
        patterns: Single pattern or list of patterns to match.
        recursive: Whether to search recursively in subdirectories.

    Returns:
        Path to the first matching file, or None if not found.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    directory = Path(directory)

    for pattern in patterns:
        if recursive:
            matches = list(directory.rglob(pattern))
        else:
            matches = list(directory.glob(pattern))

        if matches:
            # Return the first match
            return matches[0]

    return None

def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create: bool = False
) -> Tuple[bool, str]:
    """
    Validate a filesystem path.

    Args:
        path: The path to validate.
        must_exist: If True, the path must exist.
        must_be_file: If True, the path must be a file.
        must_be_dir: If True, the path must be a directory.
        create: If True, create the directory if it doesn't exist (only for directories).

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)

    if must_exist and not path.exists():
        if create and not must_be_file:
            try:
                path.mkdir(parents=True, exist_ok=True)
                return True, ""
            except Exception as e:
                return False, f"Could not create directory {path}: {str(e)}"
        return False, f"Path does not exist: {path}"

    if must_be_file and not path.is_file():
        return False, f"Path is not a file: {path}"

    if must_be_dir and not path.is_dir():
        return False, f"Path is not a directory: {path}"

    return True, ""
