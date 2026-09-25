"""Physics-informed Roost variants for alloy density prediction."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Subset

from aviary.core import Normalizer
from aviary.networks import ResidualNetwork
from aviary.roost.data import CompositionData
from aviary.roost.model import Roost
from aviary.utils import (
    initialize_losses,
    initialize_model,
    initialize_optim,
    init_normalizers,
)

from features import (
    GlobalPhysicalFeature,
    PhysicalFeatureScaler,
    get_global_physical_feature,
)
from models import (
    LOSS_DICT,
    PROJECT_ROOT,
    TARGET_NAME,
    TASK_DICT,
    _default_roost_model_params,
    _default_roost_setup_params,
    _ensemble_prediction_stats,
    _fit_target_normalizer,
    _run_roost_fit,
    regression_metrics,
    save_roost_checkpoint,
)


class PhysicsTrunkRoost(Roost):
    """Roost with one normalized global physical feature concatenated before trunk_nn."""

    def __init__(self, *args: Any, n_global_features: int = 1, **kwargs: Any) -> None:
        n_global_features = int(kwargs.pop("n_global_features", n_global_features))
        super().__init__(*args, **kwargs)
        elem_fea_len = self.model_params["elem_fea_len"]
        out_hidden = self.model_params["out_hidden"]
        trunk_hidden = self.model_params["trunk_hidden"]
        self.n_global_features = n_global_features
        self.trunk_nn = ResidualNetwork(
            elem_fea_len + n_global_features,
            out_hidden[0],
            trunk_hidden,
        )
        self.model_params["n_global_features"] = n_global_features

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
        global_phys: Tensor,
    ) -> tuple[Tensor, ...]:
        elem_fea = self.elem_embedding(elem_fea)
        crys_fea = self.material_nn(
            elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx
        )
        crys_fea = torch.cat([crys_fea, global_phys], dim=1)
        crys_fea = F.relu(self.trunk_nn(crys_fea))
        return tuple(output_nn(crys_fea) for output_nn in self.output_nns)


class PhysicsTrunkCompositionData(CompositionData):
    """CompositionData that attaches one precomputed global physical feature per row."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        global_features: np.ndarray,
    ) -> None:
        super().__init__(df=df, task_dict=task_dict)
        self.global_features = np.asarray(global_features, dtype=float).reshape(-1)

    @cache  # noqa: B019
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        composition = row[self.inputs]
        material_ids = row[self.identifiers].to_list()

        from pymatgen.core import Composition

        comp_dict = Composition(composition).fractional_composition
        weights = list(comp_dict.values())
        weights_arr = np.atleast_2d(weights).T / np.sum(weights)
        elem_fea = [elem.Z for elem in comp_dict]

        n_elems = len(comp_dict)
        self_idx = []
        nbr_idx = []
        for elem_idx in range(n_elems):
            self_idx += [elem_idx] * n_elems
            nbr_idx += list(range(n_elems))

        elem_weights = Tensor(weights_arr)
        elem_fea_tensor = LongTensor(elem_fea)
        self_idx_tensor = LongTensor(self_idx)
        nbr_idx_tensor = LongTensor(nbr_idx)
        global_phys = Tensor([self.global_features[idx]])

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(Tensor([row[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(LongTensor([row[target]]))

        return (
            (elem_weights, elem_fea_tensor, self_idx_tensor, nbr_idx_tensor, global_phys),
            targets,
            *material_ids,
        )


def collate_batch_physics_trunk(samples: tuple) -> tuple[Any, ...]:
    """Collate PhysicsTrunkCompositionData samples for batched training."""
    batch_elem_weights = []
    batch_elem_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    batch_global_phys = []
    crystal_elem_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for idx, (inputs, target, *cry_ids) in enumerate(samples):
        elem_weights, elem_fea, self_idx, nbr_idx, global_phys = inputs
        n_sites = elem_fea.shape[0]

        batch_elem_weights.append(elem_weights)
        batch_elem_fea.append(elem_fea)
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)
        batch_global_phys.append(global_phys)
        crystal_elem_idx.append(torch.tensor([idx] * n_sites))
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)
        cry_base_idx += n_sites

    return (
        (
            torch.cat(batch_elem_weights, dim=0),
            torch.cat(batch_elem_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_elem_idx),
            torch.stack(batch_global_phys),
        ),
        tuple(
            torch.stack(b_target, dim=0) for b_target in zip(*batch_targets, strict=False)
        ),
        *zip(*batch_cry_ids, strict=False),
    )


def _make_physics_trunk_loaders(
    dataset: PhysicsTrunkCompositionData,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "collate_fn": collate_batch_physics_trunk,
    }
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_kwargs)
    eval_kwargs = {
        **loader_kwargs,
        "shuffle": False,
        "batch_size": max(batch_size, 64),
    }
    val_loader = DataLoader(Subset(dataset, val_idx), **eval_kwargs)
    test_loader = DataLoader(Subset(dataset, test_idx), **eval_kwargs)
    return train_loader, val_loader, test_loader


def predict_physics_trunk_roost(
    model: PhysicsTrunkRoost,
    data_loader: DataLoader,
    normalizer_dict: dict[str, Normalizer | None],
) -> tuple[np.ndarray, np.ndarray]:
    targets, preds, _ = model.predict(data_loader)
    normalizer = normalizer_dict[TARGET_NAME]
    assert normalizer is not None
    y_true = targets[0]
    y_pred = normalizer.denorm(preds[0].squeeze()).cpu().numpy()
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def save_physics_trunk_checkpoint(
    model: PhysicsTrunkRoost,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    normalizer_dict: dict[str, Normalizer | None],
    physical_scaler: PhysicalFeatureScaler,
    global_feature: GlobalPhysicalFeature,
    model_dir: Path | str,
    run_id: int = 0,
) -> Path:
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir_path / f"checkpoint-r{run_id}.pth.tar"
    checkpoint = {
        "model_params": model.model_params,
        "state_dict": model.state_dict(),
        "epoch": model.epoch,
        "best_val_score": model.best_val_scores,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "normalizer_dict": {
            task: normalizer.state_dict() if isinstance(normalizer, Normalizer) else None
            for task, normalizer in normalizer_dict.items()
        },
        "physical_scaler": physical_scaler.state_dict(),
        "global_feature": global_feature,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def train_roost_physics_trunk(
    df: pd.DataFrame,
    composition_df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    global_feature: GlobalPhysicalFeature,
    model_dir: Path | str | None = None,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
    device: str | None = None,
    run_id: int = 0,
    verbose: bool = True,
    history_path: Path | str | None = None,
) -> dict[str, Any]:
    """Train PhysicsTrunkRoost with train-fit normalization of one global feature."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir_path = Path(model_dir) if model_dir is not None else PROJECT_ROOT / "models" / "roost_physics_trunk"
    history_path_obj = Path(history_path) if history_path is not None else None

    raw_feature = get_global_physical_feature(composition_df, global_feature)
    physical_scaler = PhysicalFeatureScaler()
    physical_scaler.fit(raw_feature[train_idx])
    normalized_feature = physical_scaler.transform(raw_feature)

    dataset = PhysicsTrunkCompositionData(
        df=df,
        task_dict=TASK_DICT,
        global_features=normalized_feature,
    )
    train_loader, val_loader, test_loader = _make_physics_trunk_loaders(
        dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=batch_size,
    )

    model_params = _default_roost_model_params(n_targets=dataset.n_targets, device=device)
    model_params["n_global_features"] = 1
    setup_params = _default_roost_setup_params(
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    model = initialize_model(
        model_class=PhysicsTrunkRoost,
        model_params=model_params,
        device=device,
    )
    optimizer, scheduler = initialize_optim(model, **setup_params)
    loss_func_dict = initialize_losses(TASK_DICT, LOSS_DICT, robust=False)
    normalizer_dict = init_normalizers(TASK_DICT, device=device)
    _fit_target_normalizer(dataset, train_idx, normalizer_dict)

    if verbose:
        print(
            f"Training PhysicsTrunkRoost ({global_feature}) on {device} "
            f"for up to {epochs} epochs"
        )

    training_history = _run_roost_fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer_dict=normalizer_dict,
        epochs=epochs,
        patience=patience,
        run_id=run_id,
        verbose=verbose,
        model_name=f"roost_physics_trunk_{global_feature}",
        history_path=history_path_obj,
    )

    checkpoint_path = save_physics_trunk_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer_dict=normalizer_dict,
        physical_scaler=physical_scaler,
        global_feature=global_feature,
        model_dir=model_dir_path,
        run_id=run_id,
    )

    y_test_true, y_test_pred = predict_physics_trunk_roost(
        model, test_loader, normalizer_dict
    )
    test_metrics = regression_metrics(y_test_true, y_test_pred)
    y_val_true, y_val_pred = predict_physics_trunk_roost(
        model, val_loader, normalizer_dict
    )
    val_metrics = regression_metrics(y_val_true, y_val_pred)

    train_loader_eval = DataLoader(
        Subset(dataset, train_idx),
        shuffle=False,
        batch_size=max(batch_size, 64),
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch_physics_trunk,
    )
    y_train_true, y_train_pred = predict_physics_trunk_roost(
        model, train_loader_eval, normalizer_dict
    )
    train_metrics = regression_metrics(y_train_true, y_train_pred)

    if verbose:
        print(
            f"PhysicsTrunk train metrics: MAE={train_metrics['MAE']:.4f}, "
            f"RMSE={train_metrics['RMSE']:.4f}, R2={train_metrics['R2']:.4f}"
        )
        print(
            f"PhysicsTrunk validation metrics: MAE={val_metrics['MAE']:.4f}, "
            f"RMSE={val_metrics['RMSE']:.4f}, R2={val_metrics['R2']:.4f}"
        )
        print(
            f"PhysicsTrunk test metrics: MAE={test_metrics['MAE']:.4f}, "
            f"RMSE={test_metrics['RMSE']:.4f}, R2={test_metrics['R2']:.4f}"
        )

    return {
        "model": model,
        "normalizer_dict": normalizer_dict,
        "physical_scaler": physical_scaler,
        "global_feature": global_feature,
        "checkpoint_path": checkpoint_path,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_train_true": y_train_true,
        "y_train_pred": y_train_pred,
        "y_val_true": y_val_true,
        "y_val_pred": y_val_pred,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
        "training_history": training_history,
        "history_path": history_path_obj,
    }


def train_predict_roost_ensemble_physics_trunk(
    df: pd.DataFrame,
    composition_df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    y_train: list[float] | np.ndarray,
    y_test: list[float] | np.ndarray,
    global_feature: GlobalPhysicalFeature,
    model_dir: Path | str,
    n_members: int = 5,
    base_seed: int = 0,
    split_seed: int | None = None,
    on_member_start: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train a PhysicsTrunkRoost ensemble for one global physical feature."""
    model_dir_path = Path(model_dir)
    split_label = split_seed if split_seed is not None else "unknown"

    train_preds = []
    test_preds = []
    checkpoint_paths: list[Path] = []

    for member_idx in range(n_members):
        member_seed = base_seed + member_idx
        if on_member_start is not None:
            on_member_start(member_idx, n_members, member_seed)

        torch.manual_seed(member_seed)
        np.random.seed(member_seed)

        member_dir = model_dir_path / f"split{split_label}" / f"member{member_idx}"
        member_kwargs = {"verbose": False, **kwargs}
        results = train_roost_physics_trunk(
            df=df,
            composition_df=composition_df,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            global_feature=global_feature,
            model_dir=member_dir,
            run_id=member_idx,
            **member_kwargs,
        )
        checkpoint_paths.append(results["checkpoint_path"])
        train_preds.append(results["y_train_pred"])
        test_preds.append(results["y_pred"])

    train_member_preds = np.asarray(train_preds, dtype=float)
    test_member_preds = np.asarray(test_preds, dtype=float)
    train_stats = _ensemble_prediction_stats(train_member_preds)
    test_stats = _ensemble_prediction_stats(test_member_preds)

    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    return {
        "global_feature": global_feature,
        "train_metrics": regression_metrics(y_train_arr, train_stats["y_pred_mean"]),
        "test_metrics": regression_metrics(y_test_arr, test_stats["y_pred_mean"]),
        "y_train_true": y_train_arr,
        "y_train_pred": train_stats["y_pred_mean"],
        "y_train_pred_std": train_stats["y_pred_std"],
        "y_train_confidence": train_stats["confidence"],
        "y_test_true": y_test_arr,
        "y_test_pred": test_stats["y_pred_mean"],
        "y_test_pred_std": test_stats["y_pred_std"],
        "y_test_confidence": test_stats["confidence"],
        "member_test_preds": test_member_preds,
        "checkpoint_paths": checkpoint_paths,
    }
