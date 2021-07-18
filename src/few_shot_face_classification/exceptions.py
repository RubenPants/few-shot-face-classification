"""Exception classes."""
from pathlib import Path
from typing import Optional, Union


class NoFaceException(Exception):
    """Exception thrown when no face is recognised."""
    
    def __init__(self, path: Optional[Union[str, Path]] = None):
        msg = "No face recognised!"
        if path is not None:
            msg += f" ({path})"
        super(NoFaceException, self).__init__(msg)


class MultipleFaceException(Exception):
    """Exception thrown when multiple faces are recognised."""
    
    def __init__(self, path: Optional[Union[str, Path]] = None):
        msg = "Multiple faces recognised!"
        if path is not None:
            msg += f" ({path})"
        super(MultipleFaceException, self).__init__(msg)


class InvalidImageException(Exception):
    """Image is invalid."""
    
    def __init__(self, path: Optional[Union[str, Path]] = None):
        msg = "Invalid image!"
        if path is not None:
            msg += f" ({path})"
        super(InvalidImageException, self).__init__(msg)
