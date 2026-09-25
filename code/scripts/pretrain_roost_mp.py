"""Regenerate the Phase 4 post-fix MP-pretrained Roost checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from data_loader import load_mp_density_data
from models import DEFAULT_ROOST_MP_PRETRAIN_DIR, pretrain_roost, records_to_dataframe

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_roost_mp_pretrain"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_ROOST_MP_PRETRAIN_DIR)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    best_val_score = {
        str(key): float(value)
        for key, value in dict(checkpoint.get("best_val_score", {}) or {}).items()
    }
    return {
        "checkpoint_path": _path_label(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_val_score": best_val_score,
        "best_val_score_nonempty": bool(best_val_score),
    }


def _load_records(max_records: int | None, sample_seed: int) -> list[dict[str, Any]]:
    records = load_mp_density_data()
    if max_records is None or max_records >= len(records):
        return records
    rng = np.random.default_rng(sample_seed)
    indices = np.sort(rng.choice(len(records), size=max_records, replace=False))
    return [records[int(index)] for index in indices]


def _split_indices(
    n_records: int,
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    indices = np.arange(n_records)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=random_state,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
    )
    return list(train_idx), list(val_idx), list(test_idx)


def _write_report(
    output_dir: Path,
    metrics: dict[str, dict[str, float]],
    summary: dict[str, Any],
) -> None:
    metric_rows = []
    for split, values in metrics.items():
        metric_rows.append(
            {
                "split": split,
                **values,
            }
        )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "mp_pretrain_metrics.csv", index=False)
    table_header = "| split | MAE | RMSE | R2 |"
    table_separator = "| --- | ---: | ---: | ---: |"
    table_rows = [
        f"| {row.split} | {row.MAE:.4f} | {row.RMSE:.4f} | {row.R2:.4f} |"
        for row in metrics_df.itertuples(index=False)
    ]

    lines = [
        "# Phase 4 MP Roost Pretrain Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Run Summary",
        "",
        f"- Records: {summary['n_records']:,}",
        f"- Split sizes: train={summary['n_train']:,}, validation={summary['n_val']:,}, test={summary['n_test']:,}",
        f"- Epochs/patience requested: {summary['epochs']}/{summary['patience']}",
        f"- Batch size: {summary['batch_size']}",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        f"- Checkpoint epoch: {summary['checkpoint_epoch']}",
        f"- Best validation score non-empty: {summary['best_val_score_nonempty']}",
        "",
        "## Metrics",
        "",
        table_header,
        table_separator,
        *table_rows,
        "",
    ]
    (output_dir / "phase4_roost_mp_pretrain_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if args.history_path is None:
        args.history_path = args.output_dir / "mp_pretrain_history.csv"
    args.history_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.model_dir / f"checkpoint-r{args.run_id}.pth.tar"
    if checkpoint_path.exists() and not args.force:
        metadata = _checkpoint_metadata(checkpoint_path)
        summary = {
            "status": "skipped_existing_checkpoint",
            "model_dir": _path_label(args.model_dir),
            "history_path": _path_label(args.history_path),
            "n_records": None,
            "checkpoint_path": metadata["checkpoint_path"],
            **metadata,
        }
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return

    records = _load_records(args.max_records, args.sample_seed)
    df = records_to_dataframe(records)
    train_idx, val_idx, test_idx = _split_indices(
        n_records=len(records),
        random_state=args.random_state,
    )

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    result = pretrain_roost(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=args.device,
        run_id=args.run_id,
        verbose=args.verbose,
        history_path=args.history_path,
    )

    checkpoint_path = Path(result["checkpoint_path"])
    metadata = _checkpoint_metadata(checkpoint_path)
    if not metadata["best_val_score_nonempty"]:
        raise RuntimeError(f"Checkpoint has empty best_val_score: {checkpoint_path}")

    metrics = {
        "train": result["train_metrics"],
        "validation": result["val_metrics"],
        "test": result["test_metrics"],
    }
    summary = {
        "status": "trained",
        "results_dir": _path_label(args.output_dir),
        "model_dir": _path_label(args.model_dir),
        "history_path": _path_label(args.history_path),
        "n_records": len(records),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "random_state": args.random_state,
        "max_records": args.max_records,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        **metadata,
    }
    _write_report(args.output_dir, metrics, summary)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
