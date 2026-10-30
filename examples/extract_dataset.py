"""Extract one JSONL manifest with one model initialization per process.

Each input line needs an ``image`` path and may include ``id``, ``label``, or
other metadata. Existing output files are never overwritten.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np

from rot_features.inference import ImageFeatureExtractor


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(value):
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--mode", choices=("historical", "corrected"), default="historical")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    if not rows:
        parser.error("The manifest must contain at least one image")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not isinstance(row.get("image"), str):
            parser.error(f"Manifest row {index} needs an image path string")
        if not Path(row["image"]).is_file():
            parser.error(f"Image does not exist in manifest row {index}: {row['image']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or meta_path.exists():
        parser.error("Output or metadata already exists; choose a new output path")
    started = time.time()
    metadata = {
        "status": "RUNNING", "geometry_mode": args.mode, "device": args.device,
        "cpu_threads": args.cpu_threads, "rows": len(rows), "manifest_sha256": checksum(args.manifest),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in
                         ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "segment-anything")},
        "checkpoint_sha256": {name: checksum(getattr(args, name + "_checkpoint")) for name in ("sam", "yolo", "resnet")},
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        extractor = ImageFeatureExtractor(sam_checkpoint=args.sam_checkpoint, yolo_checkpoint=args.yolo_checkpoint,
                                          resnet_checkpoint=args.resnet_checkpoint, device=args.device,
                                          cpu_threads=args.cpu_threads, mode=args.mode)
    except Exception as error:
        metadata.update(status="INITIALIZATION_ERROR", error_type=type(error).__name__, error=str(error))
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    import torch
    metadata["torch_cuda_version"] = torch.version.cuda
    metadata["cudnn_version"] = torch.backends.cudnn.version()
    if args.device.startswith("cuda"):
        metadata["gpu"] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()
    failures = incomplete = 0
    with args.output.open("x") as stream:
        for index, row in enumerate(rows):
            item_started = time.perf_counter()
            result = {key: value for key, value in row.items() if key != "reference_features"}
            result["input_sha256"] = checksum(row["image"])
            try:
                result["features"] = {key: portable(value) for key, value in extractor.extract(row["image"]).items()}
                result["status"] = "INCOMPLETE_FEATURES" if any(v is None for v in result["features"].values()) else "PASS"
                incomplete += result["status"] != "PASS"
            except Exception as error:
                result.update(status="ERROR", error_type=type(error).__name__, error=str(error))
                failures += 1
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            result["seconds"] = time.perf_counter() - item_started
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                print(json.dumps({"completed": index + 1, "total": len(rows), "errors": failures,
                                  "incomplete": incomplete, "elapsed_seconds": time.time() - started}), flush=True)
    metadata.update(status="PASS" if failures + incomplete == 0 else "INCOMPLETE", errors=failures,
                    incomplete_features=incomplete, elapsed_seconds=time.time() - started,
                    peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    output_sha256=checksum(args.output))
    if args.device.startswith("cuda"):
        metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return 0 if metadata["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
