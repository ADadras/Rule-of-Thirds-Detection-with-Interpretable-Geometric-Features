import numpy as np
import pytest

from rot_features import load_model, load_table, verify_artifacts
from rot_features.experiments import refit, replay


def test_artifacts_and_original_predictions():
    assert len(verify_artifacts()) == 5
    report = replay()
    assert report["status"] == "PASS"
    assert report["datasets"]["rotds"]["samples"] == 728
    assert report["datasets"]["rotds"]["accuracy"] == pytest.approx(0.7431318681318682)
    assert report["datasets"]["kupcp"]["accuracy"] == pytest.approx(0.9318181818181818)
    assert report["datasets"]["avt"]["accuracy"] == pytest.approx(0.7560538116591928)
    assert report["datasets"]["avt"]["samples"] == 1115


@pytest.mark.parametrize("dataset, rows", [("rotds", 3638), ("kupcp", 3516), ("avt", 1115)])
def test_feature_table_identity(dataset, rows):
    table = load_table(dataset)
    assert len(table) == rows
    assert table.source_row.is_unique
    assert not table.isna().any().any()
    assert set(table.label.unique()) == {False, True}


@pytest.mark.parametrize("dataset", ["rotds", "kupcp"])
def test_refits_reconstruct_the_original_models(dataset):
    report = refit(dataset)
    assert report["status"] == "PASS"
    assert report["test"]["matches_original_predictions"]
    if dataset == "rotds":
        assert report["matches_recorded_cv_folds"]
        assert report["refitted_cv_f1_mean"] == pytest.approx(0.7438540322556871)


def test_feature_validation_and_probabilities():
    table = load_table("rotds").head(5)
    model = load_model("rotds")
    probability = model.predict_proba(table)
    assert np.allclose(probability.sum(axis=1), 1)
    assert ((0 <= probability) & (probability <= 1)).all()
    with pytest.raises(ValueError, match="Missing features"):
        model.predict(table.drop(columns=model.input_features[0]))
    x = table[model.input_features].to_numpy(dtype=float)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict(x)
