from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import numpy as np
import mendeleev
import math

Avogadro_number = 6.02214076e23
pm_to_cm = 1e-10

from data_loader import load_miracle_radius_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_EMBEDDING_PATH = (
    PROJECT_ROOT / "data" / "features" / "embeddings" / "element" / "physical2.json"
)
DEFAULT_MENDELEEV_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "atomic_features_mendeleev.csv"

GlobalPhysicalFeature = Literal["density_mixing_rule", "density_atoms"]
PackingEfficiencyMethod = Literal["median", "mae_opt"]

def is_valid_element(element: str) -> bool:
    """
    Check if an element is valid.
    """
    try:
        mendeleev.element(element)
        return True
    except:
        return False

def density_of_element(element: str) -> float:
    """
    Get the density of an element in g/cm^3.
    """
    try:
        density = mendeleev.element(element).density
        return density
    except:
        raise ValueError(f"Element {element} not found in mendeleev.")

def mass_of_element(element: str) -> float:
    """
    Calculate the mass of an element.
    """
    try:
        mass = mendeleev.element(element).atomic_weight
        return mass
    except:
        raise ValueError(f"Element {element} not found in mendeleev.")

def radius_of_element_miracle_2013(element: str) -> float:
    """
    Get the radius of an element from Miracle 2013.

    Args:
        element: the element symbol

    Returns:
        The radius of the element in pm
    """
    df_miracle_radius = load_miracle_radius_data(year="2013")
    try:
        radius = df_miracle_radius.loc[df_miracle_radius["Element"] == element, "Radius"].values[0]
        return radius
    except:
        raise ValueError(f"Element {element} not found in Miracle 2013.")

def radius_of_element_miracle_2011(element: str) -> float:
    """
    Get the radius of an element from Miracle 2011.

    Args:
        element: the element symbol

    Returns:
        The radius of the element in pm
    """
    df_miracle_radius = load_miracle_radius_data(year="2011")
    try:
        radius = df_miracle_radius.loc[df_miracle_radius["Element"] == element, "Radius"].values[0]
        return radius
    except:
        raise ValueError(f"Element {element} not found in Miracle 2011.")

def radius_of_element_miracle_2015(element: str) -> float:
    """
    Get the radius of an element from Miracle 2015.

    Args:
        element: the element symbol

    Returns:
        The radius of the element in pm
    """
    df_miracle_radius = load_miracle_radius_data(year="2015")
    try:
        radius = df_miracle_radius.loc[df_miracle_radius["Element"] == element, "Radius"].values[0]
        return radius
    except:
        raise ValueError(f"Element {element} not found in Miracle 2015.")



def radius_of_element_mendeleev(element: str) -> float:
    """
    Get the radius of an element from Mendeleev.

    Args:
        element: the element symbol

    Returns:
        The radius of the element in pm
    """
    try:
        radius = mendeleev.element(element).atomic_radius
        return radius
    except:
        raise ValueError(f"Element {element} not found in mendeleev.")

def elements_composition_features(composition: list[dict[str, float]]) -> pd.DataFrame:
    """
    Generate elements composition features from composition list.

    Args:
        composition: list of composition dictionaries, each dictionary is a composition of an alloy

    Returns:
        DataFrame of elements composition features, each row is an alloy, each column is an element
    
    Example:
    >>> composition = [{'Zr': 0.41, 'Ti': 0.14, 'Cu': 0.125, 'Ni': 0.1, 'Be': 0.225}, {'Zr': 0.4675, 'Ti': 0.0825, 'Cu': 0.075, 'Ni': 0.1, 'Be': 0.275}]
    >>> elements_composition_features(composition)
       Zr  Ti  Cu  Ni  Be
    0  0.41 0.14 0.125 0.1 0.225
    1  0.4675 0.0825 0.075 0.1 0.275

    """
    df_composition = pd.json_normalize(composition).fillna(0.0)
    return df_composition


def mendeleev_composition_features(
    composition: list[dict[str, float]],
    feature_path: Path | str = DEFAULT_MENDELEEV_FEATURE_PATH,
) -> pd.DataFrame:
    """Aggregate local Mendeleev elemental properties over alloy compositions.

    Features are atomic-fraction-weighted statistics computed from
    `data/features/atomic_features_mendeleev.csv`.
    """
    df_composition = elements_composition_features(composition)
    feature_table = pd.read_csv(feature_path).set_index("Element")
    property_columns = ["AtomicNumber", "Radius", "Density", "AtomicMass"]
    missing_properties = set(property_columns) - set(feature_table.columns)
    if missing_properties:
        raise ValueError(f"Mendeleev feature table is missing columns: {sorted(missing_properties)}")

    rows: list[dict[str, float]] = []
    for _, row in df_composition.iterrows():
        positive = row[row > 0.0]
        missing_elements = [element for element in positive.index if element not in feature_table.index]
        if missing_elements:
            raise ValueError(f"Mendeleev feature table is missing elements: {missing_elements}")
        if positive.empty:
            raise ValueError("Composition rows must contain at least one positive element fraction.")

        weights = positive.to_numpy(dtype=float)
        weights = weights / weights.sum()
        features: dict[str, float] = {
            "n_elements": float(len(positive)),
            "max_fraction": float(weights.max()),
            "composition_entropy": float(-np.sum(weights * np.log(weights))),
        }
        element_features = feature_table.loc[positive.index, property_columns].astype(float)
        for property_name in property_columns:
            values = element_features[property_name].to_numpy(dtype=float)
            mean = float(np.sum(weights * values))
            variance = float(np.sum(weights * (values - mean) ** 2))
            features[f"{property_name}_weighted_mean"] = mean
            features[f"{property_name}_weighted_std"] = float(np.sqrt(variance))
            features[f"{property_name}_min"] = float(values.min())
            features[f"{property_name}_max"] = float(values.max())
            features[f"{property_name}_range"] = float(values.max() - values.min())
        rows.append(features)
    return pd.DataFrame(rows, index=df_composition.index)

def calulate_atoms_density(df_composition: pd.DataFrame | list[dict[str, float]], radius_of_element: callable=radius_of_element_miracle_2013) -> list[float]:
    """
    Calculate the atoms density of an alloy.

    Args:
        df_composition: DataFrame of composition, each column is an element or list of composition dictionaries
        radius_of_element: callable to get the radius of an element

    Returns:
        List of atoms density of each alloy
    """
    if isinstance(df_composition, list):
        try:
            df_composition = elements_composition_features(df_composition)
        except:
            raise ValueError(f"df_composition error.")
    total_mass = np.zeros(len(df_composition))
    for col in df_composition.columns:
        if is_valid_element(col):
            total_mass += mass_of_element(col) * df_composition[col].values
    atoms_volume = np.zeros(len(df_composition))
    for col in df_composition.columns:
        if is_valid_element(col):
            atoms_volume += df_composition[col].values * Avogadro_number * (4/3 * math.pi * (radius_of_element(col) * pm_to_cm)**3)
    return total_mass / atoms_volume


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median for finite positive weights."""
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cutoff = 0.5 * weights_sorted.sum()
    return float(values_sorted[np.searchsorted(np.cumsum(weights_sorted), cutoff, side="left")])


def estimate_packing_efficiency(
    df_composition: pd.DataFrame | list[dict[str, float]],
    density: list[float] | np.ndarray,
    method: PackingEfficiencyMethod = "median",
    radius_of_element: Callable[[str], float] = radius_of_element_miracle_2013,
    bounds: tuple[float, float] | None = (0.4, 0.95),
) -> float:
    """Estimate global packing efficiency from the provided rows only.

    The physical model is density = atomic_density * eta. Use this helper on the
    training split, then pass the returned eta to validation/test prediction.
    """
    atoms_density = np.asarray(
        calulate_atoms_density(df_composition, radius_of_element=radius_of_element),
        dtype=float,
    )
    return estimate_packing_efficiency_from_atoms(
        atoms_density,
        density,
        method=method,
        bounds=bounds,
    )


def estimate_packing_efficiency_from_atoms(
    atoms_density: list[float] | np.ndarray,
    density: list[float] | np.ndarray,
    method: PackingEfficiencyMethod = "median",
    bounds: tuple[float, float] | None = (0.4, 0.95),
) -> float:
    """Estimate global packing efficiency from precomputed atomic densities."""
    atoms_density = np.asarray(atoms_density, dtype=float)
    density_arr = np.asarray(density, dtype=float)
    if len(atoms_density) != len(density_arr):
        raise ValueError("Atomic-density and density arrays must have the same length.")
    if np.any(atoms_density <= 0) or not np.all(np.isfinite(atoms_density)):
        raise ValueError("Atomic densities must be finite and positive.")
    if not np.all(np.isfinite(density_arr)):
        raise ValueError("Densities must be finite.")

    per_alloy_eta = density_arr / atoms_density
    if method == "median":
        return float(np.median(per_alloy_eta))
    if method == "mae_opt":
        eta = _weighted_median(per_alloy_eta, atoms_density)
        if bounds is not None:
            eta = float(np.clip(eta, bounds[0], bounds[1]))
        return eta
    raise ValueError(f"Unknown packing-efficiency method: {method}")


def calculate_density_packing_efficiency(
    df_composition: pd.DataFrame | list[dict[str, float]],
    packing_efficiency: float,
    radius_of_element: Callable[[str], float] = radius_of_element_miracle_2013,
) -> list[float]:
    """Predict density using density = atomic_density * packing_efficiency."""
    atoms_density = np.asarray(
        calulate_atoms_density(df_composition, radius_of_element=radius_of_element),
        dtype=float,
    )
    return (atoms_density * float(packing_efficiency)).tolist()


def calculate_density_mixing_rule(df_comp: pd.DataFrame | list[dict[str, float]]) -> list[float]:
    """
    Calculate the density of an alloy using the mixing rule.

    Args:
        df_comp: DataFrame of composition, each column is an element or list of composition dictionaries

    Returns:
        List of density of each alloy
    """
    if isinstance(df_comp, list):
        try:
            df_comp = elements_composition_features(df_comp)
        except:
            raise ValueError(f"df_comp error.")
    columns = df_comp.columns.tolist()
    mass = np.zeros(len(df_comp))
    vol = np.zeros(len(df_comp))
    for col in columns:
        if is_valid_element(col):
            tmp_mass = mass_of_element(col) * df_comp[col].values
            mass += tmp_mass
            vol  += tmp_mass / density_of_element(col)
    density_mixing_rule = mass / vol
    return density_mixing_rule


def physical_density_features(df_composition: pd.DataFrame) -> pd.DataFrame:
    """Return mixing-rule and atoms-density columns for each alloy row."""
    mixing = np.asarray(calculate_density_mixing_rule(df_composition), dtype=float)
    atoms = np.asarray(calulate_atoms_density(df_composition), dtype=float)
    return pd.DataFrame(
        {
            "density_mixing_rule": mixing,
            "density_atoms": atoms,
        },
        index=df_composition.index,
    )


def get_global_physical_feature(
    df_composition: pd.DataFrame,
    feature: GlobalPhysicalFeature,
) -> np.ndarray:
    """Return a single global physical feature column as a 1-D array."""
    return physical_density_features(df_composition)[feature].to_numpy(dtype=float)


class PhysicalFeatureScaler:
    """Z-score scaler for a single global physical input feature."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, values: np.ndarray | list[float]) -> PhysicalFeatureScaler:
        arr = np.asarray(values, dtype=float).reshape(-1)
        self.mean_ = float(arr.mean())
        self.std_ = float(arr.std(ddof=0))
        if self.std_ == 0.0:
            self.std_ = 1.0
        return self

    def transform(self, values: np.ndarray | list[float]) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PhysicalFeatureScaler is not fitted.")
        arr = np.asarray(values, dtype=float).reshape(-1)
        return (arr - self.mean_) / self.std_

    def fit_transform(self, values: np.ndarray | list[float]) -> np.ndarray:
        return self.fit(values).transform(values)

    def state_dict(self) -> dict[str, float]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PhysicalFeatureScaler is not fitted.")
        return {"mean": self.mean_, "std": self.std_}

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> PhysicalFeatureScaler:
        scaler = cls()
        scaler.mean_ = float(state["mean"])
        scaler.std_ = float(state["std"])
        return scaler


def build_physical_element_embedding_json(
    output_path: Path | str | None = None,
    year: str = "2013",
) -> Path:
    """Build a 2-D Aviary element embedding JSON: [radius_z, mass_z] per element."""
    output = Path(output_path) if output_path is not None else DEFAULT_PHYSICAL_EMBEDDING_PATH
    df_radius = load_miracle_radius_data(year=year)
    elements = df_radius["Element"].tolist()
    radii = df_radius["Radius"].to_numpy(dtype=float)
    masses = np.array([mass_of_element(el) for el in elements], dtype=float)

    radius_z = (radii - radii.mean()) / radii.std(ddof=0)
    mass_z = (masses - masses.mean()) / masses.std(ddof=0)

    embedding = {
        element: [float(radius_z[i]), float(mass_z[i])]
        for i, element in enumerate(elements)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(embedding, handle, indent=4)
    return output


def magpie_features(composition: list[dict[str, float]]) -> pd.DataFrame:
    """
    Generate Magpie features from composition list.

    Args:
        composition: list of composition dictionaries, each dictionary is a composition of an alloy

    Returns:
        DataFrame of Magpie features, each row is an alloy, each column is a feature
    """
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core.composition import Composition
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    pymatgen_composition = [Composition(composition) for composition in composition]
    df_pymatgen_composition = pd.DataFrame({"pymatgen_composition": pymatgen_composition})
    df_magpie_features = ep_feat.featurize_dataframe(
        df_pymatgen_composition,
        col_id="pymatgen_composition",
    )
    df_magpie_features = df_magpie_features.drop(columns=["pymatgen_composition"])
    return df_magpie_features
