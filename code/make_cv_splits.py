"""Recreate the released outer random and composition-family CV folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cv_splits_recomputed.json")
    args = parser.parse_args()

    records = pd.read_csv(ROOT / "data" / "experimental_density_380.csv")
    families = pd.read_csv(ROOT / "data" / "composition_families_k36.csv")
    if not records["record_index"].equals(families["record_index"]):
        raise ValueError("Record indices in the data and family files differ")

    random_folds = [
        [train.tolist(), test.tolist()]
        for train, test in KFold(n_splits=10, shuffle=True, random_state=42).split(records)
    ]
    grouped_folds = [
        [train.tolist(), test.tolist()]
        for train, test in GroupKFold(n_splits=10).split(
            records, groups=families["cluster_kmeans_36"]
        )
    ]
    output = {
        "n_records": len(records),
        "grouped_10fold": grouped_folds,
        "random_10fold": random_folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
