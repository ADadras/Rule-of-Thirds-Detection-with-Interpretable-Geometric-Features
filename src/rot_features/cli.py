import argparse
import json
from pathlib import Path

from .experiments import refit, replay
from .frozen import verify_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the Rule-of-Thirds classification experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Verify the bundled feature tables and model states")
    historical = commands.add_parser("replay", help="Reproduce predictions on the recorded test splits")
    historical.add_argument("--output", type=Path, default=Path("outputs/replay.json"))
    train = commands.add_parser("refit", help="Refit a classifier with its published selected features")
    train.add_argument("--dataset", choices=("rotds", "kupcp"), required=True)
    train.add_argument("--output", type=Path, default=Path("outputs/refit.json"))
    image = commands.add_parser("extract", help="Extract features from an image using pretrained models")
    image.add_argument("image", type=Path)
    image.add_argument("--sam-checkpoint", type=Path, required=True)
    image.add_argument("--yolo-checkpoint", type=Path, required=True)
    image.add_argument("--resnet-checkpoint", type=Path, required=True)
    image.add_argument("--device", default="cpu")
    image.add_argument("--cpu-threads", type=int, default=2)
    image.add_argument("--mode", choices=("historical", "corrected"), default="historical")
    image.add_argument("--output", type=Path, default=Path("outputs/features.json"))
    image.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = {"status": "PASS", "verified_artifacts": verify_artifacts()}
        elif args.command == "replay":
            result = replay()
        elif args.command == "refit":
            result = refit(args.dataset)
        else:
            import math
            import numpy as np
            from .inference import ImageFeatureExtractor
            extractor = ImageFeatureExtractor(sam_checkpoint=args.sam_checkpoint, yolo_checkpoint=args.yolo_checkpoint,
                                               resnet_checkpoint=args.resnet_checkpoint, device=args.device,
                                               cpu_threads=args.cpu_threads, mode=args.mode)
            features, details = extractor.extract(args.image, return_intermediates=True)
            result = {"status": "PASS", "image": args.image.name, "geometry_mode": args.mode,
                      "features": {key: None if isinstance(value, (float, np.floating)) and not math.isfinite(value)
                                   else value.item() if isinstance(value, np.generic) else value
                                   for key, value in features.items()}}
            if any(value is None for value in result["features"].values()):
                result["status"] = "INCOMPLETE_FEATURES"
            if args.figure:
                from .visualization import save_explanation
                save_explanation(details, features, args.figure)
        if hasattr(args, "output"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "PASS" else 1
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
