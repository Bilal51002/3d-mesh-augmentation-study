import argparse
import csv
import time
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "modelnet10"
    / "original"
)

MANIFESTS = [
    (
        "train",
        PROJECT_ROOT
        / "splits"
        / "modelnet10"
        / "train.csv",
    ),
    (
        "test",
        PROJECT_ROOT
        / "splits"
        / "modelnet10"
        / "test.csv",
    ),
]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "dgnet"
)


def load_mesh(mesh_path):
    """Charge un fichier OFF sous forme de maillage Trimesh."""
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

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            "Type inattendu : {}".format(type(mesh).__name__)
        )

    return mesh


def inspect_mesh(mesh_path):
    """Vérifie la structure géométrique d'un maillage."""
    mesh = load_mesh(mesh_path)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "Forme des sommets invalide : {}".format(
                vertices.shape
            )
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "Faces non triangulaires : {}".format(
                faces.shape
            )
        )

    if len(vertices) == 0:
        raise ValueError("Aucun sommet.")

    if len(faces) == 0:
        raise ValueError("Aucune face.")

    if not np.isfinite(vertices).all():
        raise ValueError(
            "Les coordonnées contiennent NaN ou Inf."
        )

    minimum_index = int(faces.min())
    maximum_index = int(faces.max())

    if minimum_index < 0:
        raise ValueError(
            "Indice de sommet négatif : {}".format(
                minimum_index
            )
        )

    if maximum_index >= len(vertices):
        raise ValueError(
            "Indice de face hors limites : {} pour {} sommets.".format(
                maximum_index,
                len(vertices),
            )
        )

    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(
            mesh.is_winding_consistent
        ),
    }


def read_samples(limit=None):
    """Lit les manifestes train et test."""
    samples = []

    for split_name, manifest_path in MANIFESTS:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Manifeste introuvable : {}".format(
                    manifest_path
                )
            )

        with manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                sample = dict(row)
                sample["split"] = split_name
                samples.append(sample)

                if (
                    limit is not None
                    and len(samples) >= limit
                ):
                    return samples

    return samples


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit des maillages ModelNet10 avant "
            "le prétraitement DGNet."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Nombre maximal de fichiers à contrôler. "
            "Sans cette option, tous les fichiers sont analysés."
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit doit être strictement positif."
        )

    samples = read_samples(limit=args.limit)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.limit is None:
        output_name = "modelnet10_mesh_audit_full.csv"
    else:
        output_name = (
            "modelnet10_mesh_audit_first_{}.csv".format(
                args.limit
            )
        )

    output_path = OUTPUT_DIRECTORY / output_name

    fieldnames = [
        "split",
        "relative_path",
        "label",
        "class_name",
        "status",
        "vertices",
        "faces",
        "watertight",
        "winding_consistent",
        "elapsed_seconds",
        "error",
    ]

    valid_count = 0
    error_count = 0
    face_counts = []
    largest_mesh = None

    started_at = time.time()

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, sample in enumerate(samples, start=1):
            mesh_path = (
                DATASET_ROOT
                / sample["relative_path"]
            )

            sample_started_at = time.time()

            result = {
                "split": sample["split"],
                "relative_path": sample["relative_path"],
                "label": sample["label"],
                "class_name": sample["class_name"],
                "status": "ok",
                "vertices": "",
                "faces": "",
                "watertight": "",
                "winding_consistent": "",
                "elapsed_seconds": "",
                "error": "",
            }

            try:
                mesh_info = inspect_mesh(mesh_path)

                result.update(mesh_info)
                valid_count += 1

                face_counts.append(
                    mesh_info["faces"]
                )

                if (
                    largest_mesh is None
                    or mesh_info["faces"]
                    > largest_mesh["faces"]
                ):
                    largest_mesh = {
                        "path": sample["relative_path"],
                        "faces": mesh_info["faces"],
                    }

            except Exception as error:
                result["status"] = "error"
                result["error"] = str(error)
                error_count += 1

            result["elapsed_seconds"] = round(
                time.time() - sample_started_at,
                6,
            )

            writer.writerow(result)
            output_file.flush()

            print(
                "[{}/{}] {} | {} | faces={}{}".format(
                    index,
                    len(samples),
                    result["status"].upper(),
                    sample["relative_path"],
                    result["faces"],
                    (
                        ""
                        if result["status"] == "ok"
                        else " | {}".format(result["error"])
                    ),
                )
            )

    total_duration = time.time() - started_at

    print("\nRésumé de l'audit")
    print("-----------------")
    print("Fichiers contrôlés :", len(samples))
    print("Fichiers valides   :", valid_count)
    print("Fichiers en erreur :", error_count)

    if face_counts:
        print("Faces minimum      :", min(face_counts))
        print(
            "Faces moyennes     : {:.2f}".format(
                sum(face_counts) / len(face_counts)
            )
        )
        print("Faces maximum      :", max(face_counts))

    if largest_mesh is not None:
        print(
            "Maillage le plus lourd : {} ({} faces)".format(
                largest_mesh["path"],
                largest_mesh["faces"],
            )
        )

    print(
        "Durée totale       : {:.2f} secondes".format(
            total_duration
        )
    )
    print("Rapport CSV        :", output_path)

    if error_count:
        raise SystemExit(1)

    print("\nAudit terminé sans erreur.")


if __name__ == "__main__":
    main()
