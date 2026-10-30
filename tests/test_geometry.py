import numpy as np
import pytest

from rot_features import load_model
from rot_features.geometry import extract_features


def rectangle():
    mask = np.zeros((180, 300), dtype=np.uint8)
    mask[50:70, 90:110] = 1
    saliency = np.zeros_like(mask, dtype=float)
    saliency[60, 100] = 1
    return mask, saliency, [90, 50, 110, 70]


def test_normalization_on_a_rectangular_image():
    mask, saliency, box = rectangle()
    features = extract_features(mask, saliency, box, source="YOLO", confidence=0.9, mode="corrected")
    assert set(features) == set(load_model("rotds").input_features)
    assert features["bbox_x"] == pytest.approx(1 / 3)
    assert features["bbox_y"] == pytest.approx(1 / 3)
    assert features["distance_saliency_to_nearest_vertical_line"] == 0
    assert features["distance_saliency_to_nearest_horizontal_line"] == 0
    assert features["distance_mask_to_nearest_vertical_line"] == pytest.approx(0.5 / 300)
    assert features["distance_mask_to_nearest_horizontal_line"] == pytest.approx(0.5 / 180)
    assert features["average_pointwise_vertical_distance"] == pytest.approx(0.5 / 300)
    assert features["average_pointwise_horizontal_distance"] == pytest.approx(0.5 / 180)


def test_historical_coordinate_convention_is_explicit():
    mask, saliency, box = rectangle()
    features = extract_features(mask, saliency, box)
    assert features["average_pointwise_vertical_distance"] == pytest.approx(5 / 300)
    assert features["average_pointwise_horizontal_distance"] == pytest.approx(21 / 180)


def test_historical_switch_encoding_and_corrected_count():
    mask, saliency, _ = rectangle()
    mask[:] = 0
    mask[50:70, 140:160] = 1
    box = [140, 50, 160, 70]
    historical = extract_features(mask, saliency, box)
    corrected = extract_features(mask, saliency, box, mode="corrected")
    assert historical["vertical_line_switches"] == 3
    assert corrected["vertical_line_switches"] == 0


def test_empty_segmentation_is_visible():
    mask, saliency, box = rectangle()
    mask[:] = 0
    result = extract_features(mask, saliency, box)
    assert np.isnan(result["distance_mask_to_nearest_vertical_line"])
    assert np.isnan(result["average_pointwise_horizontal_distance"])


def test_invalid_geometry_is_rejected():
    mask, saliency, box = rectangle()
    with pytest.raises(ValueError, match="positive area"):
        extract_features(mask, saliency, [10, 10, 10, 20])
    saliency[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_features(mask, saliency, box)
