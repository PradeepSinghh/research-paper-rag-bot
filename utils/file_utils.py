"""
File-system utilities used across the project.

Handles safe file saving, hashing, directory management, and cleanup.
All file operations that touch disk should go through this module so
path-traversal and permission concerns are centralised.
"""

import os
import hashlib
import shutil

from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Directory Helpers ─────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create *path* (and all parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ─── File I/O ──────────────────────────────────────────────────────────────────

def save_uploaded_file(upload_dir: str, file_bytes: bytes, filename: str) -> str:
    """
    Write *file_bytes* to *upload_dir* / *filename* and return the full path.

    Security note: os.path.basename strips any directory component supplied
    by the client, preventing path-traversal attacks.

    Args:
        upload_dir:  Directory where the file will be stored.
        file_bytes:  Raw bytes from the uploaded file.
        filename:    Original filename (may contain unsafe components).

    Returns:
        Absolute path of the saved file.
    """
    ensure_dir(upload_dir)
    safe_name = os.path.basename(filename)  # strip any path components
    file_path = os.path.join(upload_dir, safe_name)

    with open(file_path, "wb") as fh:
        fh.write(file_bytes)

    logger.info(f"Saved uploaded file → {file_path} ({len(file_bytes):,} bytes)")
    return file_path


def get_file_hash(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of *file_bytes*."""
    return hashlib.sha256(file_bytes).hexdigest()


def delete_file(path: str) -> None:
    """Delete a file if it exists; log and ignore if it does not."""
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Deleted file: {path}")
    else:
        logger.debug(f"delete_file: path not found (skipping): {path}")


def delete_directory(path: str) -> None:
    """Recursively delete *path* and all its contents if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info(f"Deleted directory tree: {path}")


def file_size_mb(path: str) -> float:
    """Return the size of *path* in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)
