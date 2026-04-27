from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FEATURE_COLUMNS = [
    "PL_Index",
    "nu_syn",
    "LP_Index",
    "Pivot_Energy",
    "Frac_Variability",
    "Variability_Index",
    "nuFnu_syn",
]

LOG_SKIP_COLUMNS = {
    "Frac_Variability",
    "GLAT",
    "GLON",
    "LP_Index",
    "LP_beta",
    "PL_Index",
    "Unc_Flux1000",
    "Unc_PL_Index",
}

LOGGED_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in LOG_SKIP_COLUMNS
]

TRAIN_SAMPLES_CSV = PROJECT_ROOT / "classification_data" / "train_samples.csv"
BCU_SAMPLES_CSV = PROJECT_ROOT / "classification_data" / "bcu_samples.csv"


def build_bias_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(len(FEATURE_COLUMNS), 42),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(42, 1),
        nn.Sigmoid(),
    )


def build_greedy_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(len(FEATURE_COLUMNS), 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    checkpoint_path: Path
    qhat_path: Path
    builder: Any


MODEL_SPECS = {
    "bias_init": ModelSpec(
        key="bias_init",
        title="Bias Initialization",
        checkpoint_path=PROJECT_ROOT / "bias_init_checkpoints" / "weights_epoch_0374.pt",
        qhat_path=PROJECT_ROOT / "bias_init_outputs" / "bias_init_qhat_mccp.csv",
        builder=build_bias_model,
    ),
    "greedy_supervised": ModelSpec(
        key="greedy_supervised",
        title="Greedy Supervised",
        checkpoint_path=PROJECT_ROOT / "greedy_supervised_ckpts" / "weights_epoch_0300.pt",
        qhat_path=PROJECT_ROOT / "greedy_supervised_outputs" / "greedy_supervised_qhat_mccp.csv",
        builder=build_greedy_model,
    ),
}


def _get_nonconformity_dict(labels: list[str]) -> dict[str, Any]:
    def _make(label: str):
        return lambda row: 1.0 - row[label]

    return {label: _make(label) for label in labels}


def _testing(test_row: pd.Series, qhat_dict: dict[str, float]) -> list[str]:
    labels = list(test_row.index[1:])
    nonconf = _get_nonconformity_dict(labels)
    return [label for label, fn in nonconf.items() if fn(test_row) < qhat_dict[label]]


def _load_qhat_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, float_precision="round_trip")
    if "alpha" not in df.columns:
        raise ValueError(f"{path} must contain an 'alpha' column.")
    qhat_cols = [column for column in df.columns if column.startswith("qhat_")]
    if not qhat_cols:
        raise ValueError(f"{path} must contain one or more 'qhat_<label>' columns.")
    labels = sorted(column[len("qhat_"):] for column in qhat_cols)
    ordered = ["alpha"] + (["n_cal"] if "n_cal" in df.columns else []) + [
        f"qhat_{label}" for label in labels
    ]
    return df[ordered].sort_values("alpha").reset_index(drop=True), labels


def _select_qhat_for_alpha(
    qhat_table: pd.DataFrame,
    labels: list[str],
    alpha: float,
    *,
    allow_nearest: bool = True,
) -> tuple[dict[str, float], float]:
    alpha_values = qhat_table["alpha"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(alpha_values - alpha)))
    chosen_alpha = float(alpha_values[idx])
    if not np.isclose(chosen_alpha, alpha) and not allow_nearest:
        raise ValueError(f"Requested alpha={alpha:.6f} was not found in the qhat CSV.")
    row = qhat_table.iloc[idx]
    return {label: float(row[f"qhat_{label}"]) for label in labels}, chosen_alpha


def _inverse_transform_column(column: str, values: pd.Series | np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    if column in LOG_SKIP_COLUMNS:
        return values_np
    if column == "nuFnu_syn":
        return np.maximum(np.exp(values_np) - 1e-12, 0.0)
    if column == "nu_syn":
        return np.maximum(np.exp(values_np) - 1e12, 0.0)
    return np.maximum(np.exp(values_np), 0.0)


def _apply_log_transform_column(column: str, values: pd.Series | np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    if column in LOG_SKIP_COLUMNS:
        return values_np
    if column == "nuFnu_syn":
        return np.log(np.clip(values_np, 0.0, None) + 1e-12)
    if column == "nu_syn":
        return np.log(np.clip(values_np, 0.0, None) + 1e12)
    return np.log(np.clip(values_np, 1e-12, None))


class ModelRuntime:
    def __init__(self, spec: ModelSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        self.model = spec.builder().to(device)
        checkpoint = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.qhat_table, self.labels = _load_qhat_csv(spec.qhat_path)

    def predict(self, standardized_features: np.ndarray, alpha: float) -> dict[str, Any]:
        features_tensor = torch.tensor(standardized_features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            fsrq = self.model(features_tensor).detach().cpu().numpy().reshape(-1).astype(np.float64)
        fsrq = np.clip(fsrq, 1e-7, 1.0 - 1e-7)
        bll = 1.0 - fsrq

        qhat_dict, chosen_alpha = _select_qhat_for_alpha(
            self.qhat_table,
            self.labels,
            alpha=float(alpha),
            allow_nearest=True,
        )

        prediction_sets = []
        set_labels = []
        for bll_prob, fsrq_prob in zip(bll, fsrq):
            row = pd.Series({"true_label": "BLL", "BLL": bll_prob, "FSRQ": fsrq_prob})
            conformal_set = _testing(row, qhat_dict)
            prediction_sets.append(conformal_set)
            set_labels.append(_format_prediction_set(conformal_set))

        return {
            "bll": bll,
            "fsrq": fsrq,
            "prediction_sets": prediction_sets,
            "prediction_set_labels": set_labels,
            "alpha": float(chosen_alpha),
        }


class InferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.train_transformed = pd.read_csv(TRAIN_SAMPLES_CSV)[FEATURE_COLUMNS].astype(np.float64)
        self.bcu_transformed = pd.read_csv(BCU_SAMPLES_CSV)[FEATURE_COLUMNS].astype(np.float64)

        self.means = self.train_transformed.mean()
        self.scales = self.train_transformed.std(ddof=0).replace(0.0, 1.0)

        self.bcu_standardized = (self.bcu_transformed - self.means) / self.scales
        self.envelope_min = self.bcu_standardized.min() - 0.5
        self.envelope_max = self.bcu_standardized.max() + 0.5

        self.slider_metadata = self._build_slider_metadata()
        self.models = {
            key: ModelRuntime(spec, self.device)
            for key, spec in MODEL_SPECS.items()
        }

    def _build_slider_metadata(self) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        for column in FEATURE_COLUMNS:
            transformed_values = self.bcu_transformed[column].astype(np.float64)
            raw_values = _inverse_transform_column(column, transformed_values)
            metadata[column] = {
                "label": column,
                "logged": column in LOGGED_FEATURE_COLUMNS,
                "transformed_min": float(transformed_values.min()),
                "transformed_max": float(transformed_values.max()),
                "transformed_default": float(transformed_values.median()),
                "raw_min": float(np.min(raw_values)),
                "raw_max": float(np.max(raw_values)),
                "raw_default": float(np.median(raw_values)),
            }
        return metadata

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "app_name": "Blazar Classification",
            "description": "Classifier for Blazar Classes of Unknown type using Bias Initialization and Greedy Supervised models.",
            "feature_columns": FEATURE_COLUMNS,
            "logged_feature_columns": LOGGED_FEATURE_COLUMNS,
            "default_alpha": 0.1,
            "input_modes": [
                {"value": "auto", "label": "Auto detect"},
                {"value": "raw", "label": "Raw features"},
                {"value": "transformed", "label": "Already log-transformed"},
            ],
            "slider_features": self.slider_metadata,
            "models": [
                {
                    "key": key,
                    "title": runtime.spec.title,
                    "checkpoint": str(runtime.spec.checkpoint_path.relative_to(PROJECT_ROOT)),
                    "qhat": str(runtime.spec.qhat_path.relative_to(PROJECT_ROOT)),
                }
                for key, runtime in self.models.items()
            ],
        }

    def make_manual_feature_frame(self, transformed_values: dict[str, Any]) -> pd.DataFrame:
        row = {
            column: float(transformed_values[column])
            for column in FEATURE_COLUMNS
        }
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def parse_uploaded_file(self, filename: str, payload: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            raw_df = pd.read_csv(io.BytesIO(payload))
            extras = raw_df.copy()
            features = self._coerce_feature_frame(raw_df)
            return features, extras
        if suffix == ".npy":
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            features = self._frame_from_array(array)
            extras = pd.DataFrame({"row_id": np.arange(1, len(features) + 1)})
            return features, extras
        raise ValueError("Only .csv and .npy files are supported.")

    def predict(
        self,
        feature_frame: pd.DataFrame,
        *,
        input_mode: str = "auto",
        alpha: float = 0.1,
        extras: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        standardized_features, transformed_features, detected_mode = self._preprocess_features(
            feature_frame,
            input_mode=input_mode,
        )
        model_outputs = {
            key: runtime.predict(standardized_features, alpha)
            for key, runtime in self.models.items()
        }

        results_df = pd.DataFrame(index=np.arange(len(feature_frame)))
        if extras is not None:
            results_df = extras.reset_index(drop=True).copy()
        if results_df.empty:
            results_df["row_id"] = np.arange(1, len(feature_frame) + 1)

        for column in FEATURE_COLUMNS:
            results_df[column] = transformed_features[column].to_numpy(dtype=np.float64)

        for model_key, output in model_outputs.items():
            results_df[f"{model_key}_bll"] = output["bll"]
            results_df[f"{model_key}_fsrq"] = output["fsrq"]
            results_df[f"{model_key}_prediction_set"] = output["prediction_set_labels"]
            results_df[f"{model_key}_predicted_class"] = np.where(
                output["fsrq"] >= 0.5,
                "FSRQ",
                "BLL",
            )

        return {
            "rows": results_df.to_dict(orient="records"),
            "detected_input_mode": detected_mode,
            "alpha": float(alpha),
            "chosen_alphas": {
                key: float(output["alpha"])
                for key, output in model_outputs.items()
            },
            "batch_summary": {
                key: _summarize_model_output(output)
                for key, output in model_outputs.items()
            },
        }

    def _coerce_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if set(FEATURE_COLUMNS).issubset(frame.columns):
            selected = frame[FEATURE_COLUMNS].copy()
        else:
            if frame.shape[1] < len(FEATURE_COLUMNS):
                raise ValueError(
                    "Input must include the 7 model features in the same order as Data_Preperation.ipynb."
                )
            selected = frame.iloc[:, : len(FEATURE_COLUMNS)].copy()
            selected.columns = FEATURE_COLUMNS
        selected = selected.apply(pd.to_numeric, errors="coerce")
        if selected.isna().any().any():
            raise ValueError("Input contains missing or non-numeric feature values.")
        return selected.astype(np.float64)

    def _frame_from_array(self, array: np.ndarray) -> pd.DataFrame:
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError("NPY inputs must have shape (7,) or (n, 7).")
        return pd.DataFrame(array, columns=FEATURE_COLUMNS)

    def _apply_log_transforms(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        for column in FEATURE_COLUMNS:
            transformed[column] = _apply_log_transform_column(column, transformed[column])
        return transformed.astype(np.float64)

    def _standardize(self, transformed_frame: pd.DataFrame) -> np.ndarray:
        standardized = (transformed_frame[FEATURE_COLUMNS] - self.means) / self.scales
        return standardized.to_numpy(dtype=np.float32)

    def _score_candidate(self, standardized: np.ndarray) -> float:
        lower = self.envelope_min.to_numpy(dtype=np.float64)
        upper = self.envelope_max.to_numpy(dtype=np.float64)
        below = np.clip(lower - standardized, 0.0, None)
        above = np.clip(standardized - upper, 0.0, None)
        return float((below + above).mean())

    def _preprocess_features(
        self,
        feature_frame: pd.DataFrame,
        *,
        input_mode: str,
    ) -> tuple[np.ndarray, pd.DataFrame, str]:
        coerced = self._coerce_feature_frame(feature_frame)
        transformed_candidate = coerced.copy()
        raw_candidate = self._apply_log_transforms(coerced)

        if input_mode == "transformed":
            final_transformed = transformed_candidate
            detected_mode = "transformed"
        elif input_mode == "raw":
            final_transformed = raw_candidate
            detected_mode = "raw"
        else:
            transformed_score = self._score_candidate(self._standardize(transformed_candidate))
            raw_score = self._score_candidate(self._standardize(raw_candidate))
            if raw_score < transformed_score:
                final_transformed = raw_candidate
                detected_mode = "raw"
            else:
                final_transformed = transformed_candidate
                detected_mode = "transformed"

        standardized = self._standardize(final_transformed)
        return standardized, final_transformed, detected_mode


def _format_prediction_set(prediction_set: list[str]) -> str:
    if not prediction_set:
        return "Empty"
    if len(prediction_set) == 2:
        return "BLL, FSRQ"
    return prediction_set[0]


def _summarize_model_output(output: dict[str, Any]) -> dict[str, Any]:
    set_labels = output["prediction_set_labels"]
    fsrq = output["fsrq"]
    predicted = np.where(fsrq >= 0.5, "FSRQ", "BLL")
    return {
        "bll_predictions": int(np.sum(predicted == "BLL")),
        "fsrq_predictions": int(np.sum(predicted == "FSRQ")),
        "empty_sets": int(sum(label == "Empty" for label in set_labels)),
        "bll_only_sets": int(sum(label == "BLL" for label in set_labels)),
        "fsrq_only_sets": int(sum(label == "FSRQ" for label in set_labels)),
        "both_sets": int(sum(label == "BLL, FSRQ" for label in set_labels)),
    }


@lru_cache(maxsize=1)
def get_service() -> InferenceService:
    return InferenceService()


def metadata_payload() -> dict[str, Any]:
    return get_service().metadata_payload()


def predict_manual(transformed_values: dict[str, Any], alpha: float = 0.1) -> dict[str, Any]:
    service = get_service()
    feature_frame = service.make_manual_feature_frame(transformed_values)
    return service.predict(feature_frame, input_mode="transformed", alpha=alpha)


def predict_upload(
    filename: str,
    payload: bytes,
    *,
    input_mode: str = "auto",
    alpha: float = 0.1,
) -> dict[str, Any]:
    service = get_service()
    feature_frame, extras = service.parse_uploaded_file(filename, payload)
    return service.predict(feature_frame, input_mode=input_mode, alpha=alpha, extras=extras)
