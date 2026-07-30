import csv
import sys
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.dgnet.face_pool_compat import build_mesh_level


DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "modelnet10"
    / "original"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "splits"
    / "modelnet10"
    / "train.csv"
)


def load_triangular_mesh(mesh_path):
    """Charge un fichier OFF et vérifie que ses faces sont triangulaires."""
    loaded = trimesh.load(
        str(mesh_path),
        process=False,
        maintain_order=True,
    )

    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())

        if not geometries:
            raise RuntimeError(
                "La scène ne contient aucune géométrie."
            )

        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    faces = np.asarray(mesh.faces)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "Maillage non triangulaire : {}".format(
                faces.shape
            )
        )

    return mesh


def main():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        sample = next(reader)

    mesh_path = DATASET_ROOT / sample["relative_path"]
    mesh = load_triangular_mesh(mesh_path)

    faces = np.asarray(
        mesh.faces,
        dtype=np.int32,
    )

    print("Fichier :", mesh_path)
    print("Classe  :", sample["class_name"])
    print("Label   :", sample["label"])
    print("Sommets :", len(mesh.vertices))
    print("Faces   :", len(faces))
    print("Type des faces :", faces.dtype)
    print("Faces triangulaires :", faces.shape[1] == 3)

    levels = build_mesh_level(
        faces,
        level=6,
    )

    current_face_count = len(faces)

    print("\nPyramide DGNet :")

    for level_index, level_data in enumerate(levels):
        pool_mask, adjacency = level_data

        pool_mask = np.asarray(
            pool_mask,
            dtype=np.int32,
        )

        adjacency = np.asarray(
            adjacency,
            dtype=np.int32,
        )

        kept_faces = int(pool_mask.sum())
        missing_neighbors = int(
            (adjacency < 0).sum()
        )

        if len(pool_mask) != current_face_count:
            raise RuntimeError(
                "Niveau {} : {} masques pour {} faces.".format(
                    level_index,
                    len(pool_mask),
                    current_face_count,
                )
            )

        if adjacency.shape != (
            current_face_count,
            3,
        ):
            raise RuntimeError(
                "Adjacence incorrecte au niveau {} : {}".format(
                    level_index,
                    adjacency.shape,
                )
            )

        print(
            "Niveau {} : faces={} | conservées={} | "
            "adjacence={} | dtype={} | voisins absents={}".format(
                level_index,
                current_face_count,
                kept_faces,
                adjacency.shape,
                adjacency.dtype,
                missing_neighbors,
            )
        )

        current_face_count = kept_faces

    print(
        "\nTest réussi. Faces restantes après 6 niveaux :",
        current_face_count,
    )


if __name__ == "__main__":
    main()
