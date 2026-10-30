"""Resolve the recorded evaluation images and verify their bytes before extraction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rotds', 'kupcp', 'avt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/manifests'))
    parser.add_argument('--parts', type=int, default=1)
    args = parser.parse_args()
    if args.parts < 1:
        parser.error('--parts must be positive')
    manifest = Path(__file__).resolve().parents[1] / 'data/evaluation_images.csv'
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    resolved = []
    for row in rows:
        root = getattr(args, row['cohort']).resolve()
        image = (root / row['dataset_path']).resolve()
        if not image.is_relative_to(root):
            parser.error('A dataset path escapes its supplied root')
        if not image.is_file():
            parser.error(f'Missing image: {image}')
        if checksum(image) != row['input_sha256']:
            parser.error(f'Image checksum differs: {image}')
        resolved.append({'cohort': row['cohort'], 'source_row': int(row['source_row']),
                         'label': row['label'] == 'True', 'image': str(image)})
    args.output.mkdir(parents=True, exist_ok=True)
    destinations = [args.output / f'part_{i}.jsonl' for i in range(args.parts)]
    if any(path.exists() for path in destinations):
        parser.error('A manifest already exists; choose another output folder')
    for index, path in enumerate(destinations):
        with path.open('x') as stream:
            for row in resolved[index::args.parts]:
                stream.write(json.dumps(row) + '\n')
    print(f'PASS: verified {len(resolved)} images; wrote {args.parts} manifests')


if __name__ == '__main__':
    main()
