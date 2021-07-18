"""Check similarities between embeddings and operate accordingly."""
from pathlib import Path
from shutil import copy
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from few_shot_face_classification.utils import get_class


def get_classes(
        embs: List[np.ndarray],
        labeled_paths: List[Path],
        labeled_embs: List[np.ndarray],
        thr: float = 1.,
) -> List[Optional[str]]:
    """
    Extract the best fitting classes, None if no good match.
    
    :param embs: Embeddings to classify
    :param labeled_paths: Paths of the labeled embeddings, used to derive class from
    :param labeled_embs: Embeddings of the labeled faces
    :param thr: Distance threshold, return None if no distance falls below it
    """
    # Get all classes that belong to the labeled embeddings
    labeled_classes = [get_class(p) for p in labeled_paths]
    
    # Calculate the distance between embeddings
    dist = euclidean_distances(embs, labeled_embs)
    
    # Derive the best suiting class
    classes = []
    for d in dist:
        classes.append(
                labeled_classes[np.where(d == min(d))[0][0]]
                if min(d) <= thr
                else None
        )
    return classes


def export(
        paths: List[Path],
        embs: List[np.ndarray],
        labeled_paths: List[Path],
        labeled_embs: List[np.ndarray],
        write_f: Path,
        thr: float = 1.,
) -> None:
    """
    Export (copy) all images to their corresponding class (recognised person).
    
    :param paths: Paths of the raw images
    :param embs: Embeddings of the faces present in the raw images
    :param labeled_paths: Paths of the labeled images / faces
    :param labeled_embs: Embeddings of the corresponding labeled faces
    :param write_f: Folder to write results to (in corresponding subfolders)
    :param thr: Distance threshold
    """
    # Derive all the labeled classes
    classes = get_classes(
            embs=embs,
            labeled_paths=labeled_paths,
            labeled_embs=labeled_embs,
            thr=thr,
    )
    
    # Assign images to correct class
    for cls, path in zip(classes, paths):
        # Ignore when no class is recognised
        if cls is None:
            continue
        
        # Ensure class-folder exists and copy
        (write_f / cls).mkdir(parents=True, exist_ok=True)
        copy(path, write_f / f"{cls}/{path.name}")
