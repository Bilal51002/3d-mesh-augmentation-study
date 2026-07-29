import csv
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS: Dict[str, Path] = {
    "modelnet10": (
        PROJECT_ROOT
        / "datasets"
        / "modelnet10"
        / "original"
    ),
    "shrec16": (
        PROJECT_ROOT
        / "datasets"
        / "shrec16"
        / "original"
    ),
}


def validate_manifest(
    dataset_name: str,
    dataset_root: Path,
    split_name: str,
) -> int:
    manifest_path = (
        PROJECT_ROOT
        / "splits"
        / dataset_name
        / "{}.csv".format(split_name)
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Manifeste introuvable : {}".format(manifest_path)
        )

    valid_count = 0
    missing_paths = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            sample_path = dataset_root / row["relative_path"]

            if sample_path.is_file():
                valid_count += 1
            else:
                missing_paths.append(sample_path)

    if missing_paths:
        print(
            "\n{} / {} : {} fichier(s) introuvable(s)".format(
                dataset_name,
                split_name,
                len(missing_paths),
            )
        )

        for path in missing_paths[:20]:
            print("  - {}".format(path))

        raise RuntimeError(
            "Le manifeste contient des chemins invalides."
        )

    print(
        "{} / {} : {} chemins valides".format(
            dataset_name,
            split_name,
            valid_count,
        )
    )

    return valid_count


def main() -> None:
    total = 0

    for dataset_name, dataset_root in DATASETS.items():
        if not dataset_root.is_dir():
            raise FileNotFoundError(
                "Dataset introuvable : {}".format(dataset_root)
            )

        total += validate_manifest(
            dataset_name,
            dataset_root,
            "train",
        )

        total += validate_manifest(
            dataset_name,
            dataset_root,
            "test",
        )

    print(
        "\nValidation terminée : {} chemins corrects.".format(
            total
        )
    )


if __name__ == "__main__":
    main()  
