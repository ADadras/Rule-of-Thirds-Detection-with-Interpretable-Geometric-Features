# Feature definitions

The classifiers consume these 22 fields in the order recorded in each model’s JSON. Coordinates use `(x, y)`, with image width `W` and height `H`. Distances in x are divided by `W`; distances in y are divided by `H`. A grid intersection is called a `quadrant` in the original field names.

| Field | Meaning |
|---|---|
| `line_dist_h` | Distance from the box center to the nearest horizontal third, divided by H. |
| `line_dist_v` | Distance from the box center to the nearest vertical third, divided by W. |
| `line_switch_h` | Whether the box’s top and bottom edges have different nearest horizontal thirds. |
| `line_switch_v` | Whether the box’s left and right edges have different nearest vertical thirds. |
| `bbox_saliency_sum` | Sum of saliency inside the selected box; this field is not normalized by box area. |
| `bbox_confidence` | YOLO detection confidence, or 0.1 when SAM supplies the fallback box. |
| `bbox_contains_peak` | Whether the selected box contains the maximum of the saliency map. |
| `distance_mask_to_nearest_vertical_line` | Distance from the foreground centroid to the nearest vertical third, divided by W. |
| `distance_mask_to_nearest_horizontal_line` | Distance from the foreground centroid to the nearest horizontal third, divided by H. |
| `distance_mask_to_nearest_quadrant` | Minimum normalized Manhattan distance from the foreground centroid to a grid intersection. |
| `distance_saliency_to_nearest_vertical_line` | Distance from the saliency maximum to the nearest vertical third, divided by W. |
| `distance_saliency_to_nearest_horizontal_line` | Distance from the saliency maximum to the nearest horizontal third, divided by H. |
| `distance_saliency_to_nearest_quadrant` | Minimum normalized Manhattan distance from the saliency maximum to a grid intersection. |
| `average_pointwise_vertical_distance` | Mean distance of centerline points to a vertical third; see historical conventions below. |
| `average_pointwise_horizontal_distance` | Mean distance of centerline points to a horizontal third; see historical conventions below. |
| `vertical_line_switches` | Transitions between nearest vertical thirds along a centerline; historical encoding multiplies the count by three. |
| `horizontal_line_switches` | Transitions between nearest horizontal thirds along a centerline; historical encoding multiplies the count by three. |
| `objectness` | True when YOLO supplied the selected box; false for a SAM fallback. |
| `bbox_x` | Box center x-coordinate divided by W. |
| `bbox_y` | Box center y-coordinate divided by H. |
| `saliency_x` | Saliency-weighted centroid x-coordinate within the box, divided by W. |
| `saliency_y` | Saliency-weighted centroid y-coordinate within the box, divided by H. |

The point used by the `distance_saliency_*` fields is the map’s maximum. The `saliency_x` and `saliency_y` fields instead describe the saliency-weighted centroid within the box.

## Historical and corrected geometry

Historical mode reproduces the stored classifier inputs, including integer centerlines, a horizontal coordinate swap, and three units per recorded line transition. In that mode, the function named `draw_mask_center_vertical_line` scans columns, and the horizontal distance routine reverses the point tuple. The names alone therefore do not fully specify their historical calculations. The recovered functions and rectangular-image regression tests retain the exact definitions.

Corrected mode traces a vertical centerline through row centroids and a horizontal centerline through column centroids. It uses fractional thirds, consistent axes, and a count of one per transition. Use newly fitted models with this mode. [Reproduction notes](reproduction.md) explain the compatibility choices.

An empty foreground mask has undefined centroid and pointwise distances. The image command writes JSON nulls and `INCOMPLETE_FEATURES`; the classifiers reject nonfinite inputs.
