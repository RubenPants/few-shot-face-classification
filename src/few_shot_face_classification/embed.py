"""Methods to embed results."""
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm

from few_shot_face_classification.data import get_im_paths, load_single
from few_shot_face_classification.exceptions import MultipleFaceException, NoFaceException

# Filter out the user warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_networks() -> Tuple[MTCNN, InceptionResnetV1]:
    """Get all the networks for image detection."""
    # Create MTCNN network that extracts all potential faces from the images
    mtcnn = MTCNN(keep_all=True)
    
    # Use the VGGFace2 to create the embedding
    vggface2 = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, vggface2


def validate_face(
        im: Image,
        val_single: bool,
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
) -> bool:
    """
    Validate the image on detected faces.
    
    :param im: Image to validate
    :param val_single: Validate that strictly one image is present in the image
    :param mtcnn: MTCNN network for face extraction
    :param vggface2: VGGFace2 network to embed the face
    """
    # Create MTCNN network if not provided
    if mtcnn is None or vggface2 is None:
        mtcnn, vggface2 = get_networks()
    
    # Try to embed the face, catch exceptions if they happen
    try:
        # Check if image can be cropped
        img_cropped = mtcnn(im)
        
        # Check if strictly one face recognised
        if val_single and img_cropped.shape[0] == 0:
            print("No face")
            raise NoFaceException
        elif val_single and img_cropped.shape[0] > 1:
            print("Multi face")
            raise MultipleFaceException
        
        # Check if embedding happens correctly
        for face_arr in img_cropped:
            _ = vggface2(face_arr.unsqueeze(0)).detach().numpy()[0]
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        return False
    return True


def embed(
        im: Image,
        mtcnn: Optional[MTCNN] = None,
        vggface2: Optional[InceptionResnetV1] = None,
) -> List[np.ndarray]:
    """
    Create embeddings for every face detected by the algorithm.
    
    :param im: Image to embed
    :param mtcnn: MTCNN network for face extraction
    :param vggface2: VGGFace2 network to embed the face
    """
    # Create MTCNN network if not provided
    if mtcnn is None or vggface2 is None:
        mtcnn, vggface2 = get_networks()
    
    # Crop out the faces, return empty list if none detected
    img_cropped = mtcnn(im)
    if img_cropped is None:
        return []
    
    # Embed all detected faces
    embeddings = []
    for face_arr in img_cropped:
        embeddings.append(
                vggface2(face_arr.unsqueeze(0)).detach().numpy()[0]
        )
    return embeddings


def embed_folder(
        folder: Path,
        batch_size: int = 32,
) -> Tuple[List[Path], List[np.ndarray]]:
    """Embed all the images in the requested folder."""
    # Load in all the files to embed
    paths = get_im_paths(folder)
    
    # Split the paths into batches
    chunks = []
    for i in range(0, len(paths), batch_size):
        chunks.append(paths[i:i + batch_size])
    
    # Create embeddings for each chunk
    with Pool(cpu_count() - 2) as p:
        results = list(tqdm(p.imap(embed_batch, chunks), total=len(chunks), desc="Processing"))
    
    # Flatten out the results and return
    return [x for y in results for x in y[0]], [x for y in results for x in y[1]]


def embed_batch(
        paths: List[Path],
) -> Tuple[List[Path], List[np.ndarray]]:
    """Embed a batch of images as specified by their path, used in multiprocessing."""
    # Load in the networks
    mtcnn, vggface2 = get_networks()
    
    # Embed all the images
    return_path, return_arr = [], []
    for path in paths:
        im = load_single(path)
        emb = embed(
                im=im,
                mtcnn=mtcnn,
                vggface2=vggface2,
        )
        return_path += [path] * len(emb)
        return_arr += emb
    return return_path, return_arr
