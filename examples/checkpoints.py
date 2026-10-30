"""Verify the study's checkpoints, or explicitly download them from upstream."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verify(path, entry):
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Wrong file size: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Wrong SHA-256: {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--download", action="store_true", help="Download missing files (approximately 2.7 GB)")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "data/checkpoints.json").read_text())
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, entry in manifest.items():
        destination = args.directory / entry["filename"]
        if not destination.exists() and args.download:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with temporary.open("xb") as output:
                with urllib.request.urlopen(entry["url"], timeout=120) as response:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        output.write(block)
            verify(temporary, entry)
            temporary.rename(destination)
        if not destination.exists():
            parser.exit(1, f"Missing checkpoint: {destination}\n")
        verify(destination, entry)
        print(f"PASS {name}: {destination.name}")


if __name__ == "__main__":
    main()
