"""Compare newly extracted evaluation features with the original archived features."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rot_features.experiments import metrics
from rot_features.frozen import load_model, load_table, verify_artifacts


def compare(paths, manifest):
    verify_artifacts()
    with manifest.open(newline='') as stream:
        expected = {(r['cohort'], int(r['source_row'])): r for r in csv.DictReader(stream)}
    if not expected:
        raise ValueError('The evaluation manifest is empty')
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    identities = [(r['cohort'], r['source_row']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate image identity in the supplied runs')
    for row in rows:
        key = row['cohort'], row['source_row']
        if key not in expected:
            raise ValueError(f'Image is outside the recorded evaluation: {key}')
        if row['input_sha256'] != expected[key]['input_sha256']:
            raise ValueError(f'Image checksum differs: {key}')
        if not isinstance(row['label'], bool) or row['label'] != (expected[key]['label'] == 'True'):
            raise ValueError(f'Image label differs: {key}')
    report = {'expected_images': len(expected), 'processed_images': len(rows),
              'comparison_tolerance': {'rtol': 1e-5, 'atol': 1e-7}, 'cohorts': {}}
    for cohort in ('rotds', 'kupcp', 'avt'):
        subset = [r for r in rows if r['cohort'] == cohort]
        valid = [r for r in subset if r['status'] == 'PASS']
        item = {'processed_images': len(subset), 'complete_features': len(valid),
                'all_22_features_within_tolerance': 0, 'classifier_predictions_differing': 0,
                'status_counts': dict(Counter(r['status'] for r in subset))}
        if valid:
            model = load_model('rotds' if cohort == 'avt' else cohort)
            archived = load_table(cohort).set_index('source_row').loc[[r['source_row'] for r in valid]]
            actual = pd.DataFrame([r['features'] for r in valid])[model.input_features]
            reference = archived[model.input_features]
            difference = ~np.isclose(actual.to_numpy(dtype=float), reference.to_numpy(dtype=float), rtol=1e-5, atol=1e-7)
            prediction = model.predict(actual)
            original_prediction = model.predict(reference)
            item.update(all_22_features_within_tolerance=int((~difference.any(axis=1)).sum()),
                        classifier_predictions_differing=int((prediction != original_prediction).sum()),
                        field_counts_outside_tolerance=dict(zip(model.input_features, difference.sum(axis=0).tolist())),
                        metrics_on_recomputed_complete_features=metrics(archived.label.astype(int), prediction))
        report['cohorts'][cohort] = item
    report['pipeline_status'] = 'PASS' if len(rows) == len(expected) and all(r['status'] == 'PASS' for r in rows) else 'INCOMPLETE'
    report['feature_parity_status'] = 'PASS' if report['pipeline_status'] == 'PASS' and all(
        x['all_22_features_within_tolerance'] == x['complete_features'] for x in report['cohorts'].values()) else 'DIFFERS'
    report['prediction_parity_status'] = 'PASS' if report['pipeline_status'] == 'PASS' and all(
        x['classifier_predictions_differing'] == 0 for x in report['cohorts'].values()) else 'DIFFERS'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'data/evaluation_images.csv')
    parser.add_argument('--output', type=Path, default=Path('outputs/image_comparison.json'))
    args = parser.parse_args()
    report = compare(args.runs, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    # Feature or prediction drift is a reported scientific outcome. Incomplete
    # extraction has a nonzero exit status so it cannot masquerade as a full run.
    return 0 if report['pipeline_status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
