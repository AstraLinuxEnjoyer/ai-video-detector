"""Exercise both final objectives and comparisons on tiny real-media cohorts."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from filelock import FileLock

from training.artifact_spatial import run as spatial
from training.artifact_spatial.metadata import attach_dimensions
from training.artifact_spatial.sampling import MatchedRotation
from training.final_cohort import SPATIAL_FINAL, MOTION_FINAL
from training.final_metrics import read_json
from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.lineage import file_sha
from training.temporal_v2 import run as temporal
from training.temporal_v2.dashboard import Monitor
from training.temporal_v2.flow import rotation_for


def run_small(root, source, motion=False, coverage=False, paired=False):
    root.mkdir()
    config = read_json(source/'config.json')
    config.update(epochs=1, videos_per_epoch=8, val_videos=8, test_videos=8, val_frames=1,
                  workers=0, decode_workers=2, min_source_count=1, log_every=1)
    args = argparse.Namespace(**config)
    if coverage:
        args.previously_seen = read_json(source/'parent_best.json')['seen_ids']
    full = read_json(source/'splits.json')['splits']
    if motion:
        metadata = {}
        for line in (source/'motion_headers.jsonl').read_text().splitlines():
            try:
                entry = json.loads(line)
                metadata[entry['id']] = entry['metadata']
            except ValueError:
                pass
        eligible = read_json(source/'motion_cohort.json')['cohorts']
        for split in ('train', 'val'):
            ids = {row[0] for row in eligible[split]}
            full[split] = [dict(r, metadata=metadata[r['id']]) for r in full[split] if r['id'] in ids]
        args.val_clips = args.train_clips = 1
        args.preview_count = 1
        if paired:
            candidates = {r['id']: r for r in full['train']}
            plan = read_json(source/'paired_plan.json')['plans'][0]
            pairs = plan['pairs'][:2]
            train = [candidates[i] for pair in pairs for i in pair]
            for label in (0, 1):
                train.extend([candidates[i] for i in plan['ids'] if candidates[i]['source'] != 'DFDC'
                              and candidates[i]['label'] == label][:2])
            atomic_json(root/'previous_attempts.json', read_json(source/'previous_attempts.json'))
            plans = dict(plans=[dict(plan, pairs=pairs, ids=[r['id'] for r in train])])
            args.paired_plan_hash = fingerprint(plans)
            atomic_json(root/'paired_plan.json', dict(plans, fingerprint=args.paired_plan_hash))
        else:
            train = rotation_for(full['train'], args).take(8)
        if coverage and not paired:
            assert not {r['id'] for r in train} & set(args.previously_seen)
        val = temporal.choose_validation(full['val'], args)
        val_ids = {r['id'] for r in val}
        test = temporal.choose_validation([r for r in full['val'] if r['id'] not in val_ids], args)
    else:
        attached = attach_dimensions(full['train'], read_json(source/'dimensions.json'))
        train = MatchedRotation(attached, max_source_fraction_by_class={0:.5, 1:.5}).take(8)
        val = spatial.heldout_selection(full['val'], 8, args.seed)
        test = spatial.heldout_selection(full['test'], 8, args.seed)
        atomic_json(root/'dimensions.json', read_json(source/'dimensions.json'))
        atomic_json(root/'lineage.json', dict(seen_ids=[]))
    splits = dict(train=train, val=val, test=test, ood=[])
    payload = dict(splits=splits, fingerprint=fingerprint(splits))
    shutil.copy2(source/'initial.pt', root/'initial.pt')
    log = logger_for(root)
    if motion:
        monitor = Monitor(root)
        try:
            model, state = temporal.train(root, payload, train, val, args, log, monitor)
            initial = torch.load(root/'initial.pt', map_location='cpu', weights_only=True)
            assert not torch.equal(model.fc.weight.detach().cpu(), initial['fc.weight'])
            del model
            temporal.final_test(root, splits, args, log, monitor)
        finally:
            monitor.close()
    else:
        model, state = spatial.train(root, payload, args, log)
        initial = torch.load(root/'initial.pt', map_location='cpu', weights_only=True)
        assert not torch.equal(model.classifier[1].weight.detach().cpu(), initial['classifier.1.weight'])
        del model
        spatial.final_test(root, payload, args, log)
    assert state['complete'] and len(state['seen']) == 8
    assert read_json(root/'test_comparison.json')['treatments']['clean']['common'] > 0
    if paired:
        assert read_json(root/'coverage.json')['new_vs_previous_attempts'] >= 2
    elif coverage:
        assert read_json(root/'coverage.json')['new_to_lineage'] == 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', action='store_true')
    parser.add_argument('--paired', action='store_true')
    opts = parser.parse_args()
    deployed = [ROOT/'demo_weights'/name for name in ('spatial.pt', 'temporal.pt', 'spectral.json')]
    deployed = [p for p in deployed if p.is_file()]
    hashes = [file_sha(path) for path in deployed]
    torch.set_num_threads(2)
    with FileLock(str(ROOT/'cache/spatial_v2/training.lock'), timeout=0):
        root = Path(tempfile.mkdtemp(prefix='final_smoke_', dir=ROOT/'cache'))
        if opts.paired:
            run_small(root/'motion', MOTION_FINAL.with_name('motion_paired_dfdc_v1'),
                      motion=True, coverage=True, paired=True)
        elif opts.coverage:
            source = MOTION_FINAL.with_name('motion_coverage_v1')
            # The isolated smoke test only needs the audited header/cohort data,
            # not any output or checkpoint from the full training run.
            run_small(root/'motion', source, motion=True, coverage=True)
        else:
            run_small(root/'spatial', SPATIAL_FINAL)
            run_small(root/'motion', MOTION_FINAL, motion=True)
    assert hashes == [file_sha(path) for path in deployed]
    print('FINAL_GPU_SMOKE_OK', root)


if __name__ == '__main__':
    main()
