"""Optional image pipeline; heavyweight dependencies are imported on demand."""
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import extract_features


class ImageFeatureExtractor:
    def __init__(self, *, sam_checkpoint, yolo_checkpoint, resnet_checkpoint, device="cpu", cpu_threads=2, mode="historical"):
        for checkpoint in (sam_checkpoint, yolo_checkpoint, resnet_checkpoint):
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Required checkpoint does not exist: {checkpoint}")
        import cv2
        import torch
        from torchvision.models import resnet50
        from segment_anything import sam_model_registry, SamPredictor
        from ultralytics import YOLO
        from . import _historical_inference

        if cpu_threads < 1:
            raise ValueError("CPU thread count must be positive")
        torch.set_num_threads(cpu_threads)
        cv2.setNumThreads(1)
        self.device, self.mode = device, mode
        self.implementation = _historical_inference
        # The original experiment computed ResNet saliency on CPU.
        self.resnet = resnet50(weights=None)
        self.resnet.load_state_dict(torch.load(resnet_checkpoint, map_location="cpu", weights_only=True))
        self.resnet.eval()
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.sam = SamPredictor(sam)
        self.yolo = YOLO(str(yolo_checkpoint))

    def extract(self, image, *, return_intermediates=False):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        else:
            image = image.convert("RGB")
        saliency = self.implementation.get_gradcam(image, self.resnet)
        if not np.isfinite(saliency).all():
            raise ValueError("ResNet produced a non-finite saliency map")
        box = self.implementation.get_best_saliency_bbox(image, saliency, self.yolo, self.sam, device=self.device)
        mask = self.implementation.segment_foreground_in_bbox_sam(image, saliency, box["xyxy"], self.sam)
        result = extract_features(mask, saliency, box["xyxy"], confidence=box["score"],
                                  source=box["source"], contains_peak=box["contains_peak"], mode=self.mode)
        # Preserve the original score, whose crop clipped the final row/column.
        if self.mode == "historical":
            result["bbox_saliency_sum"] = box["saliency_sum"]
        if return_intermediates:
            return result, {"image": np.asarray(image), "saliency": saliency, "mask": mask, "box": box}
        return result
