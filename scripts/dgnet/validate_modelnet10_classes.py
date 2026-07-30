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
            "Faces non triangulaires : {}".format(
                faces.shape
            )
        )

    if len(faces) == 0:
        raise ValueError("Le maillage ne contient aucune face.")

    return mesh


def validate_sample(sample):
    mesh_path = DATASET_ROOT / sample["relative_path"]
    mesh = load_triangular_mesh(mesh_path)

    faces = np.asarray(
        mesh.faces,
        dtype=np.int32,
    )

    levels = build_mesh_level(
        faces,
        level=6,
    )

    current_faces = len(faces)
    first_missing_ratio = None

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

        if len(pool_mask) != current_faces:
            raise RuntimeError(
                "Niveau {} : {} masques pour {} faces.".format(
                    level_index,
                    len(pool_mask),
                    current_faces,
                )
            )

        if adjacency.shape != (current_faces, 3):
            raise RuntimeError(
                "Niveau {} : adjacence incorrecte {}.".format(
                    level_index,
                    adjacency.shape,
                )
            )

        kept_faces = int(pool_mask.sum())

        if kept_faces <= 0:
            raise RuntimeError(
                "Niveau {} : aucune face conservée.".format(
                    level_index
                )
            )

        if level_index == 0:
            missing_neighbors = int(
                (adjacency < 0).sum()
            )

            first_missing_ratio = (
                100.0
                * missing_neighbors
                / adjacency.size
            )

        current_faces = kept_faces

    return {
        "vertices": len(mesh.vertices),
        "initial_faces": len(faces),
        "final_faces": current_faces,
        "missing_ratio": first_missing_ratio,
        "file": sample["relative_path"],
    }


def main():
    samples_by_class = {}

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            class_name = row["class_name"]

            if class_name not in samples_by_class:
                samples_by_class[class_name] = row

    errors = []

    print(
        "Validation d'un maillage par classe ModelNet10\n"
    )

    for class_name in sorted(samples_by_class):
        sample = samples_by_class[class_name]

        try:
            result = validate_sample(sample)

            print(
                "{:<15} sommets={:<6} faces={:<6} "
                "finales={:<5} voisins_absents={:>6.2f}%".format(
                    class_name,
                    result["vertices"],
                    result["initial_faces"],
                    result["final_faces"],
                    result["missing_ratio"],
                )
            )

        except Exception as error:
            errors.append(
                (
                    class_name,
                    sample["relative_path"],
                    str(error),
                )
            )

            print(
                "{:<15} ÉCHEC : {}".format(
                    class_name,
                    error,
                )
            )

    if errors:
        print(
            "\n{} classe(s) en échec :".format(
                len(errors)
            )
        )

        for class_name, path, error in errors:
            print(
                "- {} | {} | {}".format(
                    class_name,
                    path,
                    error,
                )
            )

        raise SystemExit(1)

    print(
        "\nTest réussi : les 10 classes ont produit "
        "une pyramide DGNet à six niveaux."
    )


if __name__ == "__main__":
    main()
