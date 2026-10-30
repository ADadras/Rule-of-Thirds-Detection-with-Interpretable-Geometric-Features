"""Extract model features from a mask, a saliency map, and a bounding box."""
import numpy as np

from . import _historical_geometry as historical


def _switches(points, axis, lines):
    if not len(points):
        return 0
    nearest = np.abs(points[:, axis, None] - np.asarray(lines)).argmin(axis=1)
    return int(np.count_nonzero(nearest[1:] != nearest[:-1]))


def extract_features(mask, saliency, bbox, *, confidence=0.1, source="SAM", contains_peak=None, mode="historical"):
    """Return the 22 scalar fields consumed by the released classifiers.

    Historical mode retains integer centerlines, the horizontal coordinate swap,
    and the original three-units-per-switch encoding. Corrected mode uses
    consistent x/y coordinates and counts actual transitions. Existing fitted
    models must be used with historical features.
    """
    mask, saliency = np.asarray(mask), np.asarray(saliency, dtype=float)
    if mask.ndim != 2 or saliency.shape != mask.shape or not mask.size:
        raise ValueError("Mask and saliency must be nonempty 2D arrays of the same shape")
    if not np.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("Mask must be finite and nonnegative")
    if not np.isfinite(saliency).all() or (saliency < 0).any():
        raise ValueError("Saliency must be finite and nonnegative")
    if mode not in ("historical", "corrected"):
        raise ValueError("Unknown geometry mode")
    height, width = mask.shape
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Bounding box must have positive area within the image")
    base = historical.compute_rot_features(bbox, saliency, (width, height))
    if mode == "historical":
        thirds_x, thirds_y = historical.rule_of_thirds(width, height)
    else:
        thirds_x, thirds_y = [width / 3, width * 2 / 3], [height / 3, height * 2 / 3]
    peak = historical.find_max_saliency_point(saliency)
    centroid = historical.find_centroid(mask) if np.any(mask) else (np.nan, np.nan)
    result = {key: base[key] for key in ("line_dist_h", "line_dist_v", "line_switch_h", "line_switch_v")}
    for name, point in (("mask", centroid), ("saliency", peak)):
        result[f"distance_{name}_to_nearest_vertical_line"] = historical.get_distance_to_nearest_line(point, thirds_x, thirds_y, width, height, True)
        result[f"distance_{name}_to_nearest_horizontal_line"] = historical.get_distance_to_nearest_line(point, thirds_x, thirds_y, width, height, False)
        result[f"distance_{name}_to_nearest_quadrant"] = historical.get_distance_to_nearest_quadrant(point, thirds_x, thirds_y, width, height)
    if mode == "historical":
        vertical = historical.draw_mask_center_vertical_line(mask)
        horizontal = historical.draw_mask_center_line(mask)
        result["average_pointwise_vertical_distance"] = historical.get_pointwise_distance_to_next_line(vertical, thirds_x, thirds_y, width, height, True) if vertical else np.nan
        result["average_pointwise_horizontal_distance"] = historical.get_pointwise_distance_to_next_line(horizontal, thirds_x, thirds_y, width, height, False) if horizontal else np.nan
        # clean_table summed len(tuple), which is three for each recorded switch.
        result["vertical_line_switches"] = 3 * len(historical.track_line_switching(vertical, thirds_x, thirds_y, True))
        result["horizontal_line_switches"] = 3 * len(historical.track_line_switching(horizontal, thirds_x, thirds_y, False))
    else:
        rows = [(np.flatnonzero(mask[y] > 0).mean(), y) for y in range(height) if np.any(mask[y])]
        columns = [(x, np.flatnonzero(mask[:, x] > 0).mean()) for x in range(width) if np.any(mask[:, x])]
        rows, columns = np.asarray(rows).reshape(-1, 2), np.asarray(columns).reshape(-1, 2)
        result["average_pointwise_vertical_distance"] = float(np.abs(rows[:, 0, None] - thirds_x).min(axis=1).mean() / width) if len(rows) else np.nan
        result["average_pointwise_horizontal_distance"] = float(np.abs(columns[:, 1, None] - thirds_y).min(axis=1).mean() / height) if len(columns) else np.nan
        result["vertical_line_switches"] = _switches(rows, 0, thirds_x)
        result["horizontal_line_switches"] = _switches(columns, 1, thirds_y)
    sum_x2 = min(int(x2), width - 1) if mode == "historical" and source == "YOLO" else int(x2)
    sum_y2 = min(int(y2), height - 1) if mode == "historical" and source == "YOLO" else int(y2)
    result.update(
        bbox_saliency_sum=float(saliency[int(y1):sum_y2, int(x1):sum_x2].sum()),
        bbox_confidence=float(confidence),
        bbox_contains_peak=bool(x1 <= peak[0] <= x2 and y1 <= peak[1] <= y2) if contains_peak is None else bool(contains_peak),
        objectness=source == "YOLO", bbox_x=float(base["bbox_centroid"][0]), bbox_y=float(base["bbox_centroid"][1]),
        saliency_x=float(base["saliency_centroid"][0]), saliency_y=float(base["saliency_centroid"][1]),
    )
    return result
