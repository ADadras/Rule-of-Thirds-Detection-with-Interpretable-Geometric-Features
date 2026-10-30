"""Replay the original splits and refit the published selected models."""
import hashlib
from importlib.metadata import version

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .frozen import load_model, load_table, verify_artifacts


def metrics(target, prediction):
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
        "samples": len(target),
    }


def prediction_hash(prediction):
    return hashlib.sha256(np.asarray(prediction, dtype=np.uint8).tobytes()).hexdigest()


def replay():
    checked = verify_artifacts()
    report = {"status": "PASS", "evaluation": "Replay of the original fitted models on their recorded test rows",
              "verified_artifacts": checked, "datasets": {},
              "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")}}
    for name in ("rotds", "kupcp"):
        model = load_model(name)
        specification = model.specification
        table = load_table(name).set_index("source_row")
        test = table.loc[specification["held_out_source_rows"]]
        prediction = model.predict(test)
        observed = metrics(test.label.astype(int), prediction)
        observed["prediction_sha256"] = prediction_hash(prediction)
        if not (observed["prediction_sha256"] == specification["held_out_prediction_sha256"]
                and abs(observed["accuracy"] - specification["recorded_test_accuracy"]) < 1e-12
                and abs(observed["f1"] - specification["recorded_test_f1"]) < 1e-12):
            raise ValueError(f"Historical prediction replay differs for {name}")
        observed["recorded_cv_f1_folds"] = specification["recorded_cv_f1_folds"]
        observed["recorded_cv_f1_mean"] = specification["recorded_cv_f1_mean"]
        report["datasets"][name] = observed
    model = load_model("rotds")
    expected = model.specification["external_evaluation"]
    external = load_table("avt").set_index("source_row").loc[expected["source_rows"]]
    prediction = model.predict(external)
    observed = metrics(external.label.astype(int), prediction)
    observed["prediction_sha256"] = prediction_hash(prediction)
    if (observed["prediction_sha256"] != expected["prediction_sha256"]
            or abs(observed["accuracy"] - expected["accuracy"]) > 1e-12
            or abs(observed["f1"] - expected["f1"]) > 1e-12):
        raise ValueError("AVT prediction replay differs")
    observed["model_trained_on"] = "rotds"
    report["datasets"]["avt"] = observed
    return report


def refit(dataset):
    """Fit the published classifier with its original selected feature subset.

    This reconstructs a selected fitted model, rather than repeating the
    historical search that chose a winner using test-set F1.
    """
    model = load_model(dataset)
    specification = model.specification
    table = load_table(dataset).dropna()
    x = table[model.input_features].to_numpy(dtype=np.float64)
    target = table.label.to_numpy(dtype=int)
    train, test = train_test_split(np.arange(len(table)), test_size=0.2, random_state=42)
    if table.source_row.iloc[test].tolist() != specification["held_out_source_rows"]:
        raise ValueError("The training table no longer reconstructs the historical split")
    if dataset == "rotds":
        estimator = LogisticRegression(max_iter=5000)
    else:
        estimator = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(x[train][:, model.indices], target[train])
    prediction = pipeline.predict(x[test][:, model.indices])
    observed = metrics(target[test], prediction)
    observed["prediction_sha256"] = prediction_hash(prediction)
    observed["matches_original_predictions"] = observed["prediction_sha256"] == specification["held_out_prediction_sha256"]
    report = {"status": "PASS" if observed["matches_original_predictions"] else "DIFFERS",
              "dataset": dataset, "evaluation": "Refit with the published selected features; no new model selection",
              "selected_features": specification["selected_features"], "test": observed}
    if dataset == "rotds":
        # Original SelectKBest is fitted separately inside each validation fold.
        cv_pipeline = make_pipeline(SelectKBest(f_classif, k=x.shape[1] // 2), StandardScaler(), LogisticRegression(max_iter=5000))
        scores = cross_val_score(cv_pipeline, x[train], target[train], cv=5, scoring="f1")
        report["refitted_cv_f1_folds"] = scores.tolist()
        report["refitted_cv_f1_mean"] = float(scores.mean())
        report["matches_recorded_cv_folds"] = bool(np.allclose(scores, specification["recorded_cv_f1_folds"], atol=1e-12, rtol=0))
        if not report["matches_recorded_cv_folds"]:
            report["status"] = "DIFFERS"
    return report
