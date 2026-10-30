"""Historical definitions used to reproduce the original experiment.

See docs/reproduction.md for conventions preserved for numerical compatibility.
"""
import cv2
import numpy as np
import torch
from torchvision import transforms


def preprocess(image):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(image).unsqueeze(0)
    return (image, input_tensor)

def get_gradcam(image, resnet50):
    for param in resnet50.parameters():
        param.requires_grad = False
    target_layer = resnet50.layer4[-1]
    for param in target_layer.parameters():
        param.requires_grad = True
    image, input_tensor = preprocess(image)
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    output = resnet50(input_tensor)
    class_idx = output.argmax().item()
    resnet50.zero_grad()
    output[0, class_idx].backward()
    forward_handle.remove()
    backward_handle.remove()
    acts = activations[0].squeeze(0)
    grads = gradients[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.numpy()
    cam = cv2.resize(cam, image.size)
    return cam

def segment_foreground_in_bbox_sam(image, saliency_map, bbox_xyxy, sam_predictor, kernel_size=20):
    """
    Uses SAM to segment FG inside a bbox.
    If saliency peak is inside bbox, uses it as point seed.
    Otherwise uses bbox centroid.

    Args:
        image: HxWx3 RGB numpy array or PIL.Image
        saliency_map: HxW, normalized [0..1]
        bbox_xyxy: [x1, y1, x2, y2]
        sam_predictor: initialized SAM predictor
        kernel_size: int, morph kernel size

    Returns:
        full_mask: binary mask same size as image
    """
    if 'PIL' in str(type(image)):
        image = np.array(image)
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    crop = image[y1:y2, x1:x2]
    crop_sal = saliency_map[y1:y2, x1:x2]
    peak_y, peak_x = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    in_box = x1 <= peak_x <= x2 and y1 <= peak_y <= y2
    if in_box:
        seed_x = peak_x - x1
        seed_y = peak_y - y1
    else:
        seed_x = (x2 - x1) // 2
        seed_y = (y2 - y1) // 2
    sam_predictor.set_image(crop)
    masks, scores, logits = sam_predictor.predict(point_coords=np.array([[seed_x, seed_y]]), point_labels=[1], multimask_output=True)
    areas = [np.sum(mask) for mask in masks]
    largest_idx = np.argmax(areas)
    local_mask = masks[largest_idx].astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    smooth_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)
    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel)
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = smooth_mask
    return full_mask

def get_best_saliency_bbox(image, saliency_map, model, sam_predictor, device='cuda', primary_conf=0.5, fallback_conf=0.1, max_det=50, kernel_size=3):
    """
    Try YOLO twice:
    - If still no box, fallback: use SAM to segment at peak saliency, get bbox.

    Args:
        image: RGB numpy
        saliency_map: HxW [0..1]
        model: YOLOv8
        sam_predictor: Initialized SAM predictor
        kernel_size: smoothing for SAM mask
    """
    results = model.predict(image, device=device, conf=primary_conf, max_det=max_det, verbose=False)
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    peak_y, peak_x = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    saliency_sums = []
    contains_peak = []
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = (max(0, x1), max(0, y1))
        x2 = min(saliency_map.shape[1] - 1, x2)
        y2 = min(saliency_map.shape[0] - 1, y2)
        sal_sum = saliency_map[y1:y2, x1:x2].sum()
        saliency_sums.append(sal_sum)
        inside = x1 <= peak_x <= x2 and y1 <= peak_y <= y2
        contains_peak.append(inside)
    good_idxs = [i for i, c in enumerate(contains_peak) if c]
    if good_idxs:
        best_idx = max(good_idxs, key=lambda i: saliency_sums[i])
    else:
        results_fallback = model.predict(image, device=device, conf=fallback_conf, max_det=max_det, verbose=False)
        bboxes = results_fallback[0].boxes.xyxy.cpu().numpy()
        scores = results_fallback[0].boxes.conf.cpu().numpy()
        classes = results_fallback[0].boxes.cls.cpu().numpy()
        saliency_sums = []
        contains_peak = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = (max(0, x1), max(0, y1))
            x2 = min(saliency_map.shape[1] - 1, x2)
            y2 = min(saliency_map.shape[0] - 1, y2)
            sal_sum = saliency_map[y1:y2, x1:x2].sum()
            saliency_sums.append(sal_sum)
            inside = x1 <= peak_x <= x2 and y1 <= peak_y <= y2
            contains_peak.append(inside)
        fallback_idxs = [i for i, c in enumerate(contains_peak) if c]
        if fallback_idxs:
            best_idx = max(fallback_idxs, key=lambda i: saliency_sums[i])
        elif len(bboxes) > 0:
            best_idx = np.argmax(saliency_sums)
            print('No box contains peak, fallback to max-saliency box.')
        else:
            print('No YOLO boxes → using SAM fallback for peak saliency.')
            if 'PIL' in str(type(image)):
                image_np = np.array(image)
            else:
                image_np = image
            sam_predictor.set_image(image_np)
            masks, scores, logits = sam_predictor.predict(point_coords=np.array([[peak_x, peak_y]]), point_labels=[1], multimask_output=True)
            areas = [np.sum(mask) for mask in masks]
            largest_idx = np.argmax(areas)
            mask = masks[largest_idx].astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            coords = cv2.findNonZero(mask)
            x, y, w, h = cv2.boundingRect(coords)
            x1, y1, x2, y2 = (x, y, x + w, y + h)
            sal_sum = saliency_map[y1:y2, x1:x2].sum()
            best_box = {'xyxy': [x1, y1, x2, y2], 'score': 0.1, 'class': None, 'saliency_sum': float(sal_sum), 'contains_peak': True, 'source': 'SAM'}
            output_image = image_np.copy()
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f'SAM | Peak:True'
            cv2.putText(output_image, label, (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return best_box
    best_box_coords = bboxes[best_idx].astype(int)
    best_box = {'xyxy': best_box_coords.tolist(), 'score': float(scores[best_idx]), 'class': int(classes[best_idx]), 'saliency_sum': float(saliency_sums[best_idx]), 'contains_peak': contains_peak[best_idx], 'source': 'YOLO'}
    output_image = image.copy()
    if 'PIL' in str(type(output_image)):
        output_image = np.array(output_image)
    x1, y1, x2, y2 = best_box_coords
    cv2.rectangle(output_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    label = f"Class {best_box['class']} | Conf {best_box['score']:.2f} | Peak:{best_box['contains_peak']}"
    cv2.putText(output_image, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return best_box
