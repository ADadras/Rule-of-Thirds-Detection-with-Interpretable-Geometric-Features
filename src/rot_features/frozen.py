"""Portable inference from JSON models exported from the original fitted states."""
from importlib.resources import files
import hashlib
import io
import json

import numpy as np
import pandas as pd


def assets():
    return files("rot_features").joinpath("assets")


def artifact_manifest():
    return json.loads(assets().joinpath("manifest.json").read_text())


def load_table(dataset):
    manifest = artifact_manifest()
    if dataset not in manifest["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset}")
    content = assets().joinpath(manifest["datasets"][dataset]["path"]).read_bytes()
    return pd.read_csv(io.BytesIO(content), compression="gzip", float_precision="round_trip")


def load_model(dataset):
    manifest = artifact_manifest()
    if dataset not in manifest["models"]:
        raise ValueError(f"No released model for: {dataset}")
    return FrozenModel(json.loads(assets().joinpath(manifest["models"][dataset]["path"]).read_text()))


def verify_artifacts():
    manifest = artifact_manifest()
    checked = []
    for group in ("datasets", "models"):
        for name, entry in manifest[group].items():
            content = assets().joinpath(entry["path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {entry['path']}")
            checked.append(entry["path"])
    return checked


class FrozenModel:
    def __init__(self, specification):
        self.specification = specification
        if specification.get("schema_version") != 1 or specification.get("classes") != [0, 1]:
            raise ValueError("Unsupported model schema or class ordering")
        if specification["kind"] not in ("logistic_regression", "gradient_boosting"):
            raise ValueError("Unsupported model kind")
        self.input_features = specification["input_features"]
        self.indices = np.asarray(specification["selected_indices"], dtype=int)
        self.mean = np.asarray(specification["scaler_mean"], dtype=float)
        self.scale = np.asarray(specification["scaler_scale"], dtype=float)
        if len(self.indices) != len(self.mean) or self.mean.shape != self.scale.shape:
            raise ValueError("Inconsistent feature/scaler dimensions")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid scaler")
        if len(set(self.indices)) != len(self.indices) or (self.indices < 0).any() or (self.indices >= len(self.input_features)).any():
            raise ValueError("Invalid selected feature indices")

    def transform(self, values):
        if isinstance(values, pd.DataFrame):
            missing = set(self.input_features) - set(values)
            if missing:
                raise ValueError(f"Missing features: {sorted(missing)}")
            array = values[self.input_features].to_numpy(dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.input_features) or not np.isfinite(array).all():
            raise ValueError("Expected finite rows with all model input features")
        return (array[:, self.indices] - self.mean) / self.scale

    @staticmethod
    def _tree_prediction(tree, values):
        left = np.asarray(tree["children_left"], dtype=int)
        right = np.asarray(tree["children_right"], dtype=int)
        feature = np.asarray(tree["feature"], dtype=int)
        threshold = np.asarray(tree["threshold"], dtype=float)
        result = np.asarray(tree["value"], dtype=float)
        nodes = np.zeros(len(values), dtype=int)
        for _ in range(len(left) + 1):
            active_rows = np.flatnonzero(left[nodes] != -1)
            if not len(active_rows):
                return result[nodes]
            current = nodes[active_rows]
            go_left = values[active_rows, feature[current]] <= threshold[current]
            nodes[active_rows] = np.where(go_left, left[current], right[current])
        raise ValueError("Tree does not terminate")

    def decision_function(self, values):
        x = self.transform(values)
        specification = self.specification
        if specification["kind"] == "logistic_regression":
            return x @ np.asarray(specification["coefficients"]) + specification["intercept"]
        # sklearn's gradient-boosting predictor casts standardized inputs to float32.
        x = x.astype(np.float32)
        score = np.full(len(x), specification["initial_raw_score"], dtype=np.float64)
        for tree in specification["trees"]:
            score += specification["learning_rate"] * self._tree_prediction(tree, x)
        return score

    def predict(self, values):
        return (self.decision_function(values) > 0).astype(np.uint8)

    def predict_proba(self, values):
        probability = 1 / (1 + np.exp(-np.clip(self.decision_function(values), -709, 709)))
        return np.column_stack((1 - probability, probability))
