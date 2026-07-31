import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "dgnet"
    / "modelnet10_mesh_audit_full.csv"
)

THRESHOLDS = [
    1_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def main():
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            "Rapport introuvable : {}".format(REPORT_PATH)
        )

    rows = []

    with REPORT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["status"] != "ok":
                continue

            rows.append(
                {
                    "path": row["relative_path"],
                    "class_name": row["class_name"],
                    "split": row["split"],
                    "faces": int(row["faces"]),
                }
            )

    if not rows:
        raise RuntimeError(
            "Le rapport ne contient aucun maillage valide."
        )

    face_counts = np.asarray(
        [row["faces"] for row in rows],
        dtype=np.int64,
    )

    total = len(rows)

    print("Distribution des tailles ModelNet10")
    print("----------------------------------")
    print("Maillages :", total)
    print("Minimum   :", int(face_counts.min()))
    print("Moyenne   : {:.2f}".format(face_counts.mean()))
    print("Médiane   :", int(np.median(face_counts)))
    print("P90       :", int(np.percentile(face_counts, 90)))
    print("P95       :", int(np.percentile(face_counts, 95)))
    print("P99       :", int(np.percentile(face_counts, 99)))
    print("Maximum   :", int(face_counts.max()))

    print("\nDépassement des seuils")
    print("----------------------")

    for threshold in THRESHOLDS:
        count = int((face_counts > threshold).sum())
        percentage = 100.0 * count / total

        print(
            "> {:>7} faces : {:>4} maillages ({:>6.2f} %)".format(
                threshold,
                count,
                percentage,
            )
        )

    largest = sorted(
        rows,
        key=lambda row: row["faces"],
        reverse=True,
    )[:20]

    print("\nLes 20 maillages les plus lourds")
    print("--------------------------------")

    for index, row in enumerate(largest, start=1):
        print(
            "{:>2}. {:>7} faces | {:<12} | {}".format(
                index,
                row["faces"],
                row["class_name"],
                row["path"],
            )
        )


if __name__ == "__main__":
    main()
