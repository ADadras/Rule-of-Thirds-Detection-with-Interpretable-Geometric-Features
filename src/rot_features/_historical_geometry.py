"""Historical definitions used to reproduce the original experiment.

See docs/reproduction.md for conventions preserved for numerical compatibility.
"""
import numpy as np


def draw_mask_center_vertical_line(mask):
    """
    Draw a vertical line that traces the centroid of the mask for each column from left to right.

    Args:
        mask (numpy.ndarray): A binary mask (with values 0 and 1), where the mask represents
                               the segmented object. The object is represented by 1s and
                               the background is represented by 0s.
    Returns:
        line (list of tuples): List of (x, y) points representing the vertical center line.
    """
    height, width = mask.shape
    line = []
    for x in range(width):
        column = mask[:, x]
        non_zero_indices = np.where(column > 0)[0]
        if len(non_zero_indices) > 0:
            center_y = int(np.mean(non_zero_indices))
            line.append((x, center_y))
    return line

def draw_mask_center_line(mask):
    """
    Draw a line that traces the centroid of the mask for each row from bottom to top.

    Args:
        mask (numpy.ndarray): A binary mask (with values 0 and 1), where the mask represents
                               the segmented object. The object is represented by 1s and
                               the background is represented by 0s.
    Returns:
        line (list of tuples): List of (x, y) points representing the central line.
    """
    height, width = mask.shape
    line = []
    for y in range(height - 1, -1, -1):
        row = mask[y, :]
        non_zero_indices = np.where(row > 0)[0]
        if len(non_zero_indices) > 0:
            center_x = int(np.mean(non_zero_indices))
            line.append((center_x, y))
    line.reverse()
    return line

def rule_of_thirds(image_width, image_height):
    thirds_x = [image_width // 3, 2 * image_width // 3]
    thirds_y = [image_height // 3, 2 * image_height // 3]
    return (thirds_x, thirds_y)

def find_centroid(mask):
    coords = np.column_stack(np.where(mask > 0))
    centroid = np.mean(coords, axis=0)
    return centroid[::-1]

def find_max_saliency_point(saliency_map):
    max_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    return max_idx[::-1]

def get_pointwise_distance_to_next_line(center_line, thirds_x, thirds_y, W, H, is_vertical=True):
    """
    Computes average normalized distance of each point on the center line to the nearest RoT line.
    Vertical distances normalized by W, horizontal by H.
    """
    distances = []
    for point in center_line:
        x, y = point if is_vertical else point[::-1]
        if is_vertical:
            nearest_vertical_distance = min([abs(x - tx) for tx in thirds_x]) / W
            distances.append(nearest_vertical_distance)
        else:
            nearest_horizontal_distance = min([abs(y - ty) for ty in thirds_y]) / H
            distances.append(nearest_horizontal_distance)
    return np.mean(distances)

def get_distance_to_nearest_line(point, thirds_x, thirds_y, W, H, is_vertical=True):
    """
    Computes normalized distance from point to nearest RoT line.
    """
    x, y = point
    if is_vertical:
        return min([abs(x - tx) for tx in thirds_x]) / W
    else:
        return min([abs(y - ty) for ty in thirds_y]) / H

def get_distance_to_nearest_quadrant(centroid, thirds_x, thirds_y, W, H):
    """
    Computes normalized Manhattan distance from centroid to nearest quadrant intersection.
    """
    x, y = centroid
    quadrant_distances = [abs(x - tx) / W + abs(y - ty) / H for tx in thirds_x for ty in thirds_y]
    return min(quadrant_distances)

def track_line_switching(center_line, thirds_x, thirds_y, is_vertical=True):
    switch_points = []
    previous_closest_line = None
    for point in center_line:
        x, y = point if is_vertical else point[::-1]
        if is_vertical:
            closest_line = min([tx for tx in thirds_x], key=lambda tx: abs(x - tx))
        else:
            closest_line = min([ty for ty in thirds_y], key=lambda ty: abs(y - ty))
        if previous_closest_line and closest_line != previous_closest_line:
            if is_vertical and closest_line in thirds_x or (not is_vertical and closest_line in thirds_y):
                switch_points.append((point, previous_closest_line, closest_line))
        previous_closest_line = closest_line
    return switch_points

def compute_rot_features(bbox_xyxy, saliency_map, image_shape):
    """
    Compute RoT-based features with normalized distances.

    Args:
        bbox_xyxy: list [x1,y1,x2,y2]
        saliency_map: 2D numpy array (HxW)
        image_shape: (H, W)

    Returns:
        dict:
            line_dist_h: normalized distance centroid to nearest horizontal RoT line
            line_dist_v: normalized distance centroid to nearest vertical RoT line
            bbox_centroid: (x, y)
            bbox_centroid_norm: (x_norm, y_norm)
            saliency_centroid: (x, y)
            saliency_centroid_norm: (x_norm, y_norm)
            line_switch_h: True/False
            line_switch_v: True/False
    """
    W, H = image_shape
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cx_norm = cx / W
    cy_norm = cy / H
    thirds_x = [W / 3, 2 * W / 3]
    thirds_y = [H / 3, 2 * H / 3]
    dist_v = min((abs(cx - tx) for tx in thirds_x)) / W
    dist_h = min((abs(cy - ty) for ty in thirds_y)) / H
    left_edge = x1
    right_edge = x2
    left_nearest = thirds_x[0] if abs(left_edge - thirds_x[0]) < abs(left_edge - thirds_x[1]) else thirds_x[1]
    right_nearest = thirds_x[0] if abs(right_edge - thirds_x[0]) < abs(right_edge - thirds_x[1]) else thirds_x[1]
    line_switch_v = left_nearest != right_nearest
    top_edge = y1
    bottom_edge = y2
    top_nearest = thirds_y[0] if abs(top_edge - thirds_y[0]) < abs(top_edge - thirds_y[1]) else thirds_y[1]
    bottom_nearest = thirds_y[0] if abs(bottom_edge - thirds_y[0]) < abs(bottom_edge - thirds_y[1]) else thirds_y[1]
    line_switch_h = top_nearest != bottom_nearest
    saliency_crop = saliency_map[int(y1):int(y2), int(x1):int(x2)]
    if saliency_crop.sum() > 0:
        ys, xs = np.indices(saliency_crop.shape)
        s = saliency_crop
        cx_s = (xs * s).sum() / s.sum()
        cy_s = (ys * s).sum() / s.sum()
        cx_s += x1
        cy_s += y1
        cx_s_norm = cx_s / W
        cy_s_norm = cy_s / H
    else:
        cx_s, cy_s = (np.nan, np.nan)
        cx_s_norm, cy_s_norm = (np.nan, np.nan)
    return {'line_dist_h': dist_h, 'line_dist_v': dist_v, 'bbox_centroid': (cx_norm, cy_norm), 'saliency_centroid': (cx_s_norm, cy_s_norm), 'line_switch_h': line_switch_h, 'line_switch_v': line_switch_v}
