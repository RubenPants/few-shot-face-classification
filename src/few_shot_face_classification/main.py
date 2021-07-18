"""Complete A to Z functions on the data."""
from glob import glob
from pathlib import Path
from random import getrandbits
from shutil import move
from typing import Set

from few_shot_face_classification.data import get_im_paths, load_single
from few_shot_face_classification.embed import embed, embed_folder, get_networks, validate_face
from few_shot_face_classification.exceptions import InvalidImageException
from few_shot_face_classification.similarity import export, get_classes
from few_shot_face_classification.utils import Conflict


def recognise(
        path: Path,
        labeled_f: Path,
        thr: float = 1.,
) -> Set[str]:
    """Recognise all labeled faces present in the image, as specified by the provided path."""
    # Load in the image in which the faces are to be recognised
    im = load_single(path)
    
    # Detect faces and embed accordingly
    embs = embed(im)
    
    # Embed the data
    labeled_paths, labeled_embs = embed_folder(labeled_f)
    
    # Detect and return all classes
    classes = get_classes(
            embs=embs,
            labeled_paths=labeled_paths,
            labeled_embs=labeled_embs,
            thr=thr,
    )
    return set(classes) - {None, }


def validate_labels(
        labeled_f: Path,
        conflict: Conflict = Conflict.CRASH,
) -> None:
    """
    Validate if the labeled data is correct.
    
    :param labeled_f: Folder with labeled data
    :param conflict: How to handle conflict in the data (warn, remove, or crash execution)
    """
    # Get all image paths to validate
    paths = get_im_paths(labeled_f)
    
    # Load in networks used during validation
    mtcnn, vggface2 = get_networks()
    
    # Start validation
    for path in paths:
        im = load_single(path)
        if not validate_face(im, val_single=True, mtcnn=mtcnn, vggface2=vggface2):
            if conflict == Conflict.WARN:
                print(f"Image '{path}' is invalid!")
            elif conflict == Conflict.REMOVE:
                print(f"Invalid image '{path}', removing...")
                path.unlink(missing_ok=True)
            elif conflict == Conflict.CRASH:
                raise InvalidImageException(path)


def detect_and_export(
        raw_f: Path,
        labeled_f: Path,
        write_f: Path,
        thr: float = 1.,
        conflict: Conflict = Conflict.CRASH,
) -> None:
    """
    Detect all faces in the images and export them to the correct subfolder.
    
    :param raw_f: Folder with raw images to export / classify
    :param labeled_f: Folder with labeled images (faces)
    :param write_f: Folder to which results are written
    :param thr: Distance threshold
    :param conflict: How to handle conflict in the data (warn, remove, or crash execution)
    """
    # First, validate that all labels are indeed correct
    validate_labels(labeled_f, conflict=conflict)
    
    # Embed the data
    labeled_paths, labeled_embs = embed_folder(labeled_f)
    paths, embs = embed_folder(raw_f)
    
    # Export the results
    export(
            paths=paths,
            embs=embs,
            labeled_paths=labeled_paths,
            labeled_embs=labeled_embs,
            write_f=write_f,
            thr=thr,
    )


def add_none(
        path: Path,
        labeled_f: Path,
) -> None:
    """Add every recognised face in the image to the 'None' class in the labeled folder."""
    # Get the face extraction network
    mtcnn, _ = get_networks()
    
    # Crop the images
    im = load_single(path)
    hsh = getrandbits(128)
    _ = mtcnn(
            im,
            save_path=str(Path.cwd() / f'{hsh}.png'),
    )
    
    # Move to labeled_f
    tmp_images = glob(str(Path.cwd() / f'{hsh}*.png'))
    n = len(glob(str(labeled_f / 'none_*')))
    for i, tmp_im in enumerate(tmp_images):
        move(
                tmp_im,
                labeled_f / f'none_{n + i + 1}.png',
        )
