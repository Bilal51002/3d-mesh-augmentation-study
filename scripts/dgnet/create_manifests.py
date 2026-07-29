import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Chemin vers la racine de 3d-mesh-augmentation-study
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_class_directories(dataset_root: Path) -> List[Path]:
    """Retourne uniquement les dossiers représentant des classes."""
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir()
    )


def collect_samples(
    dataset_root: Path,
    split_name: str,
    extension: str,
    class_to_label: Dict[str, int],
) -> List[Tuple[str, int, str]]:
    """Collecte les chemins, labels numériques et noms de classes."""
    samples: List[Tuple[str, int, str]] = []

    for class_name, label in class_to_label.items():
        split_directory = dataset_root / class_name / split_name

        if not split_directory.is_dir():
            raise FileNotFoundError(
                "Dossier de split introuvable : {}".format(
                    split_directory
                )
            )

        files = sorted(
            path
            for path in split_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() == extension.lower()
        )

        if not files:
            raise RuntimeError(
                "Aucun fichier {} trouvé dans {}".format(
                    extension,
                    split_directory,
                )
            )

        for file_path in files:
            relative_path = file_path.relative_to(dataset_root)

            samples.append(
                (
                    relative_path.as_posix(),
                    label,
                    class_name,
                )
            )

    return samples


def write_manifest(
    output_path: Path,
    samples: List[Tuple[str, int, str]],
) -> None:
    """Enregistre les échantillons dans un fichier CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "relative_path",
                "label",
                "class_name",
            ]
        )

        writer.writerows(samples)


def prepare_dataset(
    dataset_name: str,
    dataset_root: Path,
    extension: str,
) -> None:
    """Génère train.csv, test.csv et classes.json."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Dataset introuvable : {}".format(dataset_root)
        )

    class_directories = get_class_directories(dataset_root)

    if not class_directories:
        raise RuntimeError(
            "Aucune classe trouvée dans {}".format(dataset_root)
        )

    class_to_label = {
        class_directory.name: index
        for index, class_directory in enumerate(class_directories)
    }

    train_samples = collect_samples(
        dataset_root=dataset_root,
        split_name="train",
        extension=extension,
        class_to_label=class_to_label,
    )

    test_samples = collect_samples(
        dataset_root=dataset_root,
        split_name="test",
        extension=extension,
        class_to_label=class_to_label,
    )

    output_directory = PROJECT_ROOT / "splits" / dataset_name

    write_manifest(
        output_directory / "train.csv",
        train_samples,
    )

    write_manifest(
        output_directory / "test.csv",
        test_samples,
    )

    with (output_directory / "classes.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_to_label,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDataset : {}".format(dataset_name))
    print("Racine  : {}".format(dataset_root))
    print("Classes : {}".format(len(class_to_label)))
    print("Train   : {}".format(len(train_samples)))
    print("Test    : {}".format(len(test_samples)))
    print("Sortie  : {}".format(output_directory))


def main() -> None:
    prepare_dataset(
        dataset_name="modelnet10",
        dataset_root=(
            PROJECT_ROOT
            / "datasets"
            / "modelnet10"
            / "original"
        ),
        extension=".off",
    )

    prepare_dataset(
        dataset_name="shrec16",
        dataset_root=(
            PROJECT_ROOT
            / "datasets"
            / "shrec16"
            / "original"
        ),
        extension=".npz",
    )

    print("\nCréation des manifestes terminée avec succès.")


if __name__ == "__main__":
    main()
