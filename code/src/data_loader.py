"""Load processed alloy density datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

DEFAULT_MERGED_DENSITY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "experimental_density_380.csv"
)

DEFAULT_MP_DENSITY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "MaterialsProject"
    / "mp_density.json"
)


def load_merged_density_data(
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Load the released experimental density records.

    Each record has:
      - composition: element symbol -> atomic fraction (float)
      - density: measured density in g/cm³ (float)

    Args:
        path: CSV file path. Defaults to data/experimental_density_380.csv.

    Returns:
        List of records in the format expected by the original training code.
    """
    data_path = Path(path) if path is not None else DEFAULT_MERGED_DENSITY_PATH
    if data_path.suffix.lower() == ".csv":
        frame = pd.read_csv(data_path)
        fraction_columns = [column for column in frame if column.startswith("x_")]
        return [
            {
                "composition": {
                    column[2:]: float(row[column])
                    for column in fraction_columns
                    if float(row[column]) > 0
                },
                "composition_str": str(row["composition_str"]),
                "density": float(row["density_g_cm3"]),
            }
            for _, row in frame.iterrows()
        ]
    with data_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_mp_density_data(
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Load Materials Project density records from JSON.

    Each record has:
      - material_id: MP identifier (optional for downstream use)
      - composition: element symbol -> atomic fraction (float)
      - composition_str: reduced formula string
      - density: DFT-calculated density in g/cm³ (float)

    Args:
        path: JSON file path. Defaults to data/raw/MaterialsProject/mp_density.json.

    Returns:
        List of records as stored in the JSON file.
    """
    data_path = Path(path) if path is not None else DEFAULT_MP_DENSITY_PATH
    with data_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_miracle_radius_data(year="2013") -> pd.DataFrame:
    """
    Load miracle radius data from CSV.

    Args:
        year: the year of the data

    Returns:
        DataFrame of miracle radius data
    """
    data_path = Path(__file__).resolve().parents[2]
    data_path = data_path / "data" / "features" / f"atomic_radii_miracle_{year}.csv"
    df = pd.read_csv(data_path)
    return df
