# Image pipeline

## Installation and checkpoints

Use Python 3.12. The CPU environment in the quick start supports artifact replay and classifier refitting. For images on Linux with CUDA 12.4, create a separate environment using the versions recorded during validation:

```bash
python3.12 -m venv .venv-image
source .venv-image/bin/activate
python -m pip install -r requirements-inference-lock.txt
python -m pip install --no-deps .
python -m pip check
```

For a different platform, `python -m pip install '.[inference,figures]'` resolves the optional dependencies. Such an environment may differ numerically from the recorded Linux/CUDA run.

The image pipeline was exercised on an NVIDIA RTX 3090 with PyTorch 2.5.1 and torchvision 0.20.1 using CUDA 12.4, Ultralytics 8.3.179, and OpenCV 4.11.0.86. SAM is pinned to source commit `dca509fe793f601edb92606367a655c15ac00fdf`. The neural networks use pretrained weights; this method trains only the downstream classifiers.

Three exact checkpoints are required:

| Component | Weights | Approximate size | Upstream source |
|---|---|---:|---|
| Saliency | ResNet50, ImageNet-1K V1 | 103 MB | [torchvision](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py) |
| Object proposals | YOLO11s | 19 MB | [Ultralytics assets](https://github.com/ultralytics/assets/releases/tag/v8.3.0) |
| Foreground segmentation | SAM ViT-H | 2.56 GB | [Segment Anything](https://github.com/facebookresearch/segment-anything) |

Each upstream model and checkpoint retains its own terms. [checkpoints.json](../data/checkpoints.json) records the exact file sizes, download URLs, and full SHA-256 hashes of the weights used for validation. Obtain and verify them with:

```bash
# Explicitly downloads approximately 2.7 GB from the upstream sources
python examples/checkpoints.py checkpoints --download

# Verify an existing set without downloading
python examples/checkpoints.py checkpoints
```

The helper does not replace an existing checkpoint. An interrupted download remains in a `.partial` file for inspection.

## One image

```bash
rot-reproduce extract path/to/image.jpg \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --yolo-checkpoint checkpoints/yolo11s.pt \
  --resnet-checkpoint checkpoints/resnet50-0676ba61.pth \
  --device cuda:0 --cpu-threads 2 \
  --output outputs/features.json --figure outputs/explanation.png
```

`--device cpu` also selects a CPU path, although SAM ViT-H is computationally expensive. The reported GPU validation uses two CPU threads per process. Grad-CAM runs on the CPU, matching the original implementation; YOLO and SAM use the selected device. The command uses RGB images and resizes the ResNet input directly to 224 × 224, then applies ImageNet normalization. It preserves the study’s preprocessing rather than substituting torchvision’s default resize-and-crop transform.

The selected box favors proposals containing the saliency maximum and then compares their saliency sums. The detector first uses confidence 0.5 and retries at 0.1; if no proposal is available, SAM provides a fallback. Foreground extraction selects the largest of SAM’s candidate masks and applies the original morphological smoothing. See the recovered functions for exact behavior.

## A collection of images

Create a JSONL file with one image path per line, plus optional identifiers and labels:

```json
{"id": "example-1", "image": "/path/to/first.jpg", "label": true}
{"id": "example-2", "image": "/path/to/second.jpg", "label": false}
```

```bash
python examples/extract_dataset.py images.jsonl outputs/features.jsonl \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --yolo-checkpoint checkpoints/yolo11s.pt \
  --resnet-checkpoint checkpoints/resnet50-0676ba61.pth \
  --device cuda:0
```

The script initializes each model once, writes and flushes one record per image, and preserves errors and undefined features explicitly. The adjacent metadata file records checkpoint hashes, dependency versions, processing time, and peak memory. It refuses to overwrite an existing run. Use separate manifests and output files to divide a collection across GPUs.

Historical mode is the default. Corrected geometry changes feature definitions and requires refitting classifiers. [Feature definitions](features.md) explain both modes.

## Recompute the recorded evaluation images

The [image catalog](../data/evaluation_images.csv) identifies the exact 2,547 evaluation images using relative paths, archived source-row identifiers, labels, and image checksums. It contains metadata only. Obtain the three original datasets, then resolve and verify their files:

```bash
python examples/prepare_evaluation.py \
  --rotds /path/to/rule_of_thirds \
  --kupcp /path/to/KU_PCP \
  --avt /path/to/avt_image_db \
  --parts 2 --output outputs/manifests
```

The expected dataset roots contain RoT-DS `positive/` and `negative/` folders, KU-PCP `train_img/` and `test_img/`, and AVT `jpg/`. The preparation command checks every image hash before writing manifests. Run the batch extractor on each manifest, selecting a separate GPU and output file for each process. A single GPU can process both parts sequentially.

```bash
# Repeat for part_1.jsonl, with another output file and device as appropriate
python examples/extract_dataset.py outputs/manifests/part_0.jsonl outputs/part_0.jsonl \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --yolo-checkpoint checkpoints/yolo11s.pt \
  --resnet-checkpoint checkpoints/resnet50-0676ba61.pth \
  --device cuda:0

python examples/compare_image_runs.py outputs/part_0.jsonl outputs/part_1.jsonl \
  --output outputs/image_comparison.json
```

The comparison verifies image identity, rejects duplicates and changed labels, then compares all features and classifier predictions with the archived values. It reports extraction coverage, feature differences, prediction differences, and newly calculated classification metrics separately. An incomplete extraction has a nonzero exit status; numerical differences are retained as explicit outcomes in the report.
