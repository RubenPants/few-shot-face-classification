"""Load in the data."""
from glob import glob
from pathlib import Path
from typing import Any, List

from PIL import ExifTags, Image

from few_shot_face_classification.utils import IMG_SUFFIX


# Load images and fix rotation if required
def _fix_rot(im: Image) -> Image:
    """Fix rotation of the PIL image."""
    # Check if rotation information is included
    if not hasattr(im, '_getexif') or not im._getexif():
        return im
    
    # Check if rotated and correct correspondingly
    exif = dict((ExifTags.TAGS[k], v) for k, v in im._getexif().items() if k in ExifTags.TAGS)
    if 'Orientation' not in exif:
        pass
    elif exif['Orientation'] == 3:
        im = im.rotate(180, expand=True)
    elif exif['Orientation'] == 6:
        im = im.rotate(270, expand=True)
    elif exif['Orientation'] == 8:
        im = im.rotate(90, expand=True)
    return im


def get_im_paths(folder: Path) -> List[Path]:
    """Get the paths to all images present in the folder."""
    return [Path(path) for path in sorted(glob(str(folder / '*'))) if Path(path).suffix in IMG_SUFFIX]


def load_single(path: Path) -> Image:
    """Load a single image."""
    return _fix_rot(Image.open(path).convert('RGB'))


def load_folder(folder: Path) -> List[Any]:
    """Load in all the image data under the requested folder."""
    # Load in all the image paths
    paths = get_im_paths(folder)
    
    # Convert to images and fix rotation
    return [load_single(path) for path in paths]
