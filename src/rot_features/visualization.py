"""Render the intermediate representations used by the image pipeline."""
from pathlib import Path

import numpy as np


def save_explanation(details, features, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image, saliency, mask = (details[key] for key in ("image", "saliency", "mask"))
    height, width = mask.shape
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), facecolor="white")
    axes = axes.ravel()
    for axis in axes[:3]:
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_title("Image and thirds", fontsize=13, color="#183348", pad=12)
    for x in (width / 3, width * 2 / 3):
        axes[0].axvline(x, color="white", linewidth=1.1, alpha=0.9)
    for y in (height / 3, height * 2 / 3):
        axes[0].axhline(y, color="white", linewidth=1.1, alpha=0.9)
    axes[1].imshow(saliency, cmap="magma", alpha=0.65, vmin=0, vmax=1)
    axes[1].set_title("ResNet50 saliency", fontsize=13, color="#183348", pad=12)
    axes[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="Blues", vmin=0, vmax=1.4, alpha=0.65)
    x1, y1, x2, y2 = details["box"]["xyxy"]
    axes[2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#f5c26b", linewidth=1.5))
    axes[2].set_title("Object and mask", fontsize=13, color="#183348", pad=12)
    keys = ["distance_saliency_to_nearest_vertical_line", "distance_saliency_to_nearest_horizontal_line",
            "distance_mask_to_nearest_vertical_line", "distance_mask_to_nearest_horizontal_line"]
    labels = ["Saliency → vertical", "Saliency → horizontal", "Mask → vertical", "Mask → horizontal"]
    values = [features[key] for key in keys]
    axes[3].barh(labels[::-1], values[::-1], color=["#365c76", "#365c76", "#bd803b", "#bd803b"])
    axes[3].set_xlim(0, max(0.34, float(np.nanmax(values)) * 1.2))
    axes[3].set_xlabel("Distance / image dimension", fontsize=10)
    axes[3].set_title("Distances to grid lines", fontsize=13, color="#183348", pad=12)
    axes[3].set_box_aspect(height / width)
    axes[3].spines[["top", "right"]].set_visible(False)
    axes[3].tick_params(labelsize=9)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.94, bottom=0.08, wspace=0.55, hspace=0.36)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)
