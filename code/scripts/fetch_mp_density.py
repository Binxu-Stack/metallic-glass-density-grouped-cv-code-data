"""Download density data for all Materials Project entries with computed density."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from mp_api.client import MPRester
from pymatgen.core.composition import Composition

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parents[1] / "data" / "raw" / "MaterialsProject"

FIELDS = [
    "material_id",
    "formula_pretty",
    "composition",
    "density",
    "elements",
    "nelements",
    "volume",
    "nsites",
    "is_stable",
]


def _require_api_key() -> None:
    if not os.environ.get("MP_API_KEY"):
        raise SystemExit(
            "MP_API_KEY is not set. Export your Materials Project API key first:\n"
            '  export MP_API_KEY="your_api_key_here"'
        )


def _atomic_write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def _atomic_write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame = pd.DataFrame(records)
    for column in ("composition", "elements"):
        if column in frame.columns:
            frame[column] = frame[column].apply(json.dumps)
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def fetch_mp_density_records(
    *,
    chunk_size: int = 1000,
    num_chunks: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch summary records from Materials Project and keep entries with density."""
    _require_api_key()

    search_criteria = {"deprecated": False}

    with MPRester(use_document_model=False) as mpr:
        total_count = mpr.materials.summary.count(search_criteria)
        print(f"Non-deprecated summary entries in MP: {total_count}")

        docs = mpr.materials.summary.search(
            fields=FIELDS,
            deprecated=False,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
        )

    print(f"Fetched {len(docs)} records from API")

    records_with_density = [
        doc for doc in docs if doc.get("density") is not None
    ]
    skipped = len(docs) - len(records_with_density)
    print(
        f"Kept {len(records_with_density)} records with density "
        f"(skipped {skipped} without density)"
    )
    return records_with_density


def to_project_compatible_record(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert an MP summary record to the project's density JSON schema."""
    composition_source = doc.get("composition") or doc.get("formula_pretty")
    if composition_source is None:
        raise ValueError(
            f"Record {doc.get('material_id')} has no composition or formula"
        )

    comp = Composition(composition_source)
    composition = {
        element: amount / comp.num_atoms
        for element, amount in comp.as_dict().items()
    }

    return {
        "material_id": doc["material_id"],
        "composition": composition,
        "composition_str": comp.reduced_formula,
        "density": float(doc["density"]),
    }


def write_raw_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    output_format: str,
) -> None:
    if output_format in {"json", "both"}:
        json_path = output_dir / "mp_density_raw.json"
        _atomic_write_json(json_path, records)
        print(f"Wrote {len(records)} raw records to {json_path}")

    if output_format in {"csv", "both"}:
        csv_path = output_dir / "mp_density_raw.csv"
        _atomic_write_csv(csv_path, records)
        print(f"Wrote {len(records)} raw records to {csv_path}")


def write_compatible_output(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    compatible_records = [to_project_compatible_record(doc) for doc in records]
    json_path = output_dir / "mp_density.json"
    _atomic_write_json(json_path, compatible_records)
    print(f"Wrote {len(compatible_records)} project-compatible records to {json_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Materials Project density data for all entries "
            "with a computed density value."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: script directory).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of records per API chunk (default: 1000).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to fetch (useful for smoke tests).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of density records to keep after filtering.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Raw output format (default: both).",
    )
    parser.add_argument(
        "--no-compat",
        action="store_true",
        help="Skip writing project-compatible mp_density.json output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero")
    if args.num_chunks is not None and args.num_chunks <= 0:
        raise SystemExit("--num-chunks must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    records = fetch_mp_density_records(
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
    )

    if args.limit is not None:
        records = records[: args.limit]
        print(f"Limited output to {len(records)} records")

    write_raw_outputs(records, args.output_dir, args.format)

    if not args.no_compat:
        write_compatible_output(records, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted before outputs were finalized.", file=sys.stderr)
        raise SystemExit(1) from None
