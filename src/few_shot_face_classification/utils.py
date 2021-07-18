"""Utilisation functions and constants."""
from enum import Enum
from pathlib import Path
from typing import Optional

# All supported image suffixes
IMG_SUFFIX = {'.png', '.jpg', '.jpeg'}


def get_class(p: Path) -> Optional[str]:
    """Get the class-name of the given path."""
    cls = p.with_suffix('').name.split('_')[0].strip()
    return None if cls.lower() == 'none' else cls


class Conflict(Enum):
    """How to handle conflicts in the data."""
    WARN = 0
    REMOVE = 1
    CRASH = 2
