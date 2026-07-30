"""
Couche de compatibilité pour le pooling officiel de DGNet.

Le noyau C++ de DGNet attend des indices int32, tandis que certaines
opérations NumPy/Jittor produisent des tableaux int64. Ce module conserve
l'algorithme officiel tout en imposant explicitement le type int32.
"""

import importlib.util
from pathlib import Path

import jittor as jt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_FACE_POOL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dgnet"
    / "jmesh"
    / "layers"
    / "face_pool.py"
)


def _load_official_face_pool():
    """Charge directement face_pool.py sans importer tout le package jmesh."""
    if not OFFICIAL_FACE_POOL_PATH.is_file():
        raise FileNotFoundError(
            "Fichier officiel introuvable : {}".format(
                OFFICIAL_FACE_POOL_PATH
            )
        )

    spec = importlib.util.spec_from_file_location(
        "dgnet_official_face_pool",
        str(OFFICIAL_FACE_POOL_PATH),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Impossible de charger {}".format(
                OFFICIAL_FACE_POOL_PATH
            )
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_OFFICIAL = _load_official_face_pool()


def face_pool_single_compat(adjacency):
    """
    Applique un niveau de pooling en garantissant le type int32.
    """
    adjacency = adjacency.int32()

    color = _OFFICIAL.map_face(
        adjacency,
        strict=True,
    ).int32()

    pool_mask = (color == 1).int32()

    new_adjacency = _OFFICIAL.update_adj(
        pool_mask,
        adjacency,
    ).int32()

    new_adjacency = new_adjacency[
        color == 1,
        :
    ].int32()

    # L'opération NumPy cumsum produit souvent du int64.
    # Nous forçons donc explicitement int32.
    pool_mask_np = np.asarray(
        pool_mask.numpy(),
        dtype=np.int32,
    )

    remap_np = (
        np.cumsum(
            pool_mask_np,
            dtype=np.int32,
        )
        * pool_mask_np
        - 1
    ).astype(np.int32)

    new_adjacency_np = np.asarray(
        new_adjacency.numpy(),
        dtype=np.int32,
    )

    missing_neighbors = new_adjacency_np == -1

    # Éviter l'indexation négative pendant le remapping.
    safe_adjacency = new_adjacency_np.copy()
    safe_adjacency[missing_neighbors] = 0

    new_adjacency_np = remap_np[safe_adjacency]
    new_adjacency_np[missing_neighbors] = -1

    new_adjacency = jt.array(
        new_adjacency_np
    ).int32()

    return pool_mask, new_adjacency


def build_mesh_level(faces, level=6):
    """
    Construit la pyramide DGNet avec des indices int32.

    Args:
        faces: tableau de forme (nombre_faces, 3).
        level: nombre de niveaux de pooling.

    Returns:
        Liste de couples [pool_mask, adjacency].
    """
    faces = np.asarray(
        faces,
        dtype=np.int32,
    )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "Les faces doivent avoir la forme (F, 3), reçu : {}".format(
                faces.shape
            )
        )

    face_adjacency_np = _OFFICIAL.build_face_adjacency(
        faces
    )

    face_adjacency_np = np.asarray(
        face_adjacency_np,
        dtype=np.int32,
    )

    face_adjacency = jt.array(
        face_adjacency_np
    ).int32()

    levels = []

    for _ in range(level):
        pool_mask, new_adjacency = face_pool_single_compat(
            face_adjacency
        )

        levels.append(
            [
                np.asarray(
                    pool_mask.numpy(),
                    dtype=np.int32,
                ),
                np.asarray(
                    face_adjacency.numpy(),
                    dtype=np.int32,
                ),
            ]
        )

        face_adjacency = new_adjacency.int32()

    return levels
