"""Paired, inference-only temporal ablations on fixed held-out clips."""
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from tqdm import tqdm

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.data import assert_disjoint, fixed_selection
from training.spatial.engine import loader, seed_all
from training.spatial.lineage import file_sha
from training.spatial.media import FrameCache
from training.spatial.metrics import grouped_report
from training.temporal_model import build_temporal_model
from .media import Clips, prepare
from .run import evaluation_coverage

VARIANTS = ('clean', 'reverse', 'shuffle_0', 'shuffle_1', 'shuffle_2',
            'freeze_first', 'freeze_middle', 'freeze_last')
DERIVED = {'static_ensemble': ('freeze_first', 'freeze_middle', 'freeze_last'),
           'shuffle_ensemble': ('shuffle_0', 'shuffle_1', 'shuffle_2')}


def temporal_variant(x, name, identities, seed):
    """Keep spatial preprocessing fixed; change only the time axis."""
    length = x.shape[2]
    if name == 'clean':
        return x
    if name == 'reverse':
        return x.flip(2)
    if name.startswith('freeze_'):
        index = {'freeze_first': 0, 'freeze_middle': length//2, 'freeze_last': length-1}[name]
        return x[:, :, index:index+1].expand_as(x).contiguous()
    if name.startswith('shuffle_'):
        orders = []
        for video_id, clip in identities:
            value = int(fingerprint([seed, name, video_id, clip])[:16], 16) % (2**63-1)
            orders.append(torch.randperm(length, generator=torch.Generator().manual_seed(value)))
        return torch.stack([sample[:, order] for sample, order in zip(x, orders)])
    raise ValueError(f'Unknown temporal variant: {name}')


def paired_summary(rows, variant, seed=42, repeats=1000):
    y = np.array([r['label'] for r in rows], dtype=int)
    clean = np.array([r['scores']['clean'] for r in rows])
    altered = np.array([r['scores'][variant] for r in rows])
    a, b = (clean >= .5) == y, (altered >= .5) == y
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['group']].append(i)
    indices = [np.array(values) for values in groups.values()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        values.append(float((a[sample].astype(float)-b[sample]).mean()))
    result = dict(n=len(rows), changed_decisions=int(((clean >= .5) != (altered >= .5)).sum()),
                  clean_correct_variant_wrong=int((a & ~b).sum()),
                  clean_wrong_variant_correct=int((~a & b).sum()),
                  clean_minus_variant_accuracy=float(a.mean()-b.mean()),
                  paired_group_bootstrap_95ci=[float(v) for v in np.quantile(values, [.025, .975])],
                  mean_abs_score_change=float(np.abs(clean-altered).mean()),
                  mean_signed_fake_score_change=float((altered-clean).mean()))
    if variant in VARIANTS:
        result['mean_abs_clip_logit_change'] = float(np.mean([
            abs(a-b) for row in rows for a, b in zip(row['logits']['clean'], row['logits'][variant])]))
    return result


def summarize(rows, requested, unsupported, seed):
    reports = {}
    for variant in (*VARIANTS, *DERIVED):
        predictions = [dict(r, score=r['scores'][variant]) for r in rows]
        report = grouped_report(predictions)
        report['coverage'] = evaluation_coverage(requested, predictions, unsupported)
        report['paired_with_clean'] = paired_summary(rows, variant, seed)
        report['by_native_fps'] = {}
        for name in ('below_16', '16_and_above'):
            selected = [r for r in predictions if ('below_16' if r['native_fps'] < 16 else '16_and_above') == name]
            if selected:
                report['by_native_fps'][name] = grouped_report(selected)
        reports[variant] = report
    return reports


def metadata_control(rows):
    """Fixed decoder-rate rule, not trained and not a model prediction."""
    rule = [dict(r, score=float(r['native_fps'] < 16)) for r in rows]
    report = grouped_report(rule)
    report.update(rule='fake iff source FPS < 16; fixed decoder rate, no threshold search',
                  caveat='This rule reads headers; the RGB model does not. Its success shows dataset '
                         'confounding, not that the model necessarily learned this rule.')
    return report


def infer(model, records, args, batch_size, log):
    dataset = Clips(records, args.image_size)
    values = defaultdict(lambda: defaultdict(list))
    logits = defaultdict(lambda: defaultdict(list))
    model.eval().requires_grad_(False)
    device = torch.device(args.device)
    begin = time.perf_counter()
    with torch.inference_mode():
        batches = loader(dataset, batch_size, args.workers)
        loop = tqdm(batches, desc='MOTION AUDIT')
        for step, (x, _, indices) in enumerate(loop, 1):
            identities = [(dataset.items[i][0]['id'], dataset.items[i][1]) for i in indices.tolist()]
            for variant in VARIANTS:
                transformed = temporal_variant(x, variant, identities, args.seed).to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    output = model(transformed).flatten().float()
                if not torch.isfinite(output).all():
                    raise RuntimeError(f'Nonfinite audit output: {variant}')
                scores = output.sigmoid().cpu().tolist()
                raw = output.cpu().tolist()
                for (video_id, clip), score, value in zip(identities, scores, raw):
                    values[video_id][variant].append((clip, score))
                    logits[video_id][variant].append((clip, value))
            if step == 1 or step % 25 == 0 or step == len(batches):
                log.info('AUDIT batches=%d/%d clips=%d/%d variants=%d elapsed=%.1fs',
                         step, len(batches), min(step*batch_size, len(dataset)), len(dataset),
                         len(VARIANTS), time.perf_counter()-begin)
    rows = []
    for record in records:
        video_id = record['id']
        per_clip = {name: [value for _, value in sorted(values[video_id][name])] for name in VARIANTS}
        if any(len(scores) != len(record['clips']) for scores in per_clip.values()):
            raise RuntimeError(f'Missing audit scores for {video_id}')
        scores = {name: float(np.mean(p)) for name, p in per_clip.items()}
        scores.update({name: float(np.mean([scores[key] for key in keys])) for name, keys in DERIVED.items()})
        rows.append(dict(id=video_id, group=record['group'], path=record['path'],
                         source=record['source'], label=record['label'], scores=scores,
                         clip_scores=per_clip,
                         logits={name: [v for _, v in sorted(logits[video_id][name])] for name in VARIANTS},
                         native_fps=record['metadata']['fps'],
                         repeated_adjacent_fraction=float(np.mean(record['repeated_adjacent_fraction']))))
    return rows


def save_predictions(path, rows):
    fields = ['id', 'group', 'source', 'label', 'path', 'native_fps', 'repeated_adjacent_fraction',
              *VARIANTS, *DERIVED, 'clip_scores', 'logits']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fields if key in row}
            output.update(row['scores'])
            for key in ('clip_scores', 'logits'):
                output[key] = json.dumps(row[key])
            writer.writerow(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'cache/temporal_v2/runs/r2plus1d_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-videos', type=int, default=800)
    parser.add_argument('--splits', nargs='+', choices=('test', 'ood'), default=['test', 'ood'])
    parser.add_argument('--inference-batch-size', type=int, default=8)
    parsed = parser.parse_args(argv)
    if parsed.max_videos < 2 or parsed.inference_batch_size < 1:
        parser.error('Positive batch size and at least two requested videos are required')
    run = parsed.run_dir.resolve()
    output = (parsed.output or ROOT/'reports'/f'temporal_motion_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    with FileLock(str(ROOT/'cache/spatial_v2/training.lock'), timeout=0), FileLock(str(run/'run.lock'), timeout=0):
        log = logger_for(output)
        args = argparse.Namespace(**json.loads((run/'config.json').read_text()))
        if getattr(args, 'input_mode', 'rgb') != 'rgb':
            raise ValueError('This audit is for the RGB baseline. Motion runs include flow controls in each epoch report.')
        seed_all(args.seed)
        payload = json.loads((run/'splits.json').read_text(encoding='utf-8'))
        if fingerprint(payload['splits']) != payload['fingerprint']:
            raise ValueError('Frozen split fingerprint mismatch')
        assert_disjoint(payload['splits'])
        hashes = {name: file_sha(run/name) for name in ('best.pt', 'last.pt', 'checkpoint.pt')}
        protocol = dict(run=str(run), weights_sha256=hashes['best.pt'], split_hash=payload['fingerprint'],
                        seed=args.seed, max_videos=parsed.max_videos, splits=parsed.splits,
                        inference_batch_size=parsed.inference_batch_size,
                        aggregation='mean probabilities across clips; static ensemble averages three anchor freezes',
                        variants=VARIANTS, derived=DERIVED,
                        caveats=['Perturbations are out of distribution; a change does not prove object-motion reasoning.',
                                 'Static ensemble uses first/middle/last frames, not every possible frame.',
                                 'No test threshold fitting, optimizer, fine-tuning or deployment change.',
                                 'Previously inspected test/OOD, not a new blind benchmark. OOD can be fake-only.',
                                 'FPS and identical-neighbor diagnostics are not semantic motion or camera tracking.'])
        atomic_json(output/'protocol.json', protocol)
        model = build_temporal_model(False).to(args.device)
        model.load_state_dict(torch.load(run/'best.pt', weights_only=True, map_location='cpu'))
        cache = FrameCache(run, args.cache_gb, args.reserve_gb)
        report = dict(protocol=protocol, splits={}, metadata_controls={})
        try:
            for split in parsed.splits:
                eligible = [r for r in payload['splits'][split] if args.sequence_fps > 0 or r.get('kind') != 'image_sequence']
                requested = fixed_selection(eligible, parsed.max_videos, args.seed)
                if not requested:
                    report['splits'][split] = dict(status='empty')
                    continue
                records = prepare(requested, cache, split, args.val_clips, args, log, output, allow_short=True)
                unsupported = json.loads((output/f'{split}_decode.json').read_text())['unsupported']
                if not records:
                    raise ValueError(f'No supported videos in {split}')
                log.info('AUDIT %s videos=%d clips=%d weights=%s', split, len(records),
                         sum(len(r['clips']) for r in records), hashes['best.pt'])
                rows = infer(model, records, args, parsed.inference_batch_size, log)
                save_predictions(output/f'{split}_predictions.csv', rows)
                report['splits'][split] = summarize(rows, requested, unsupported, args.seed)
                report['metadata_controls'][split] = metadata_control(rows)
                atomic_json(output/f'{split}_report.json', report['splits'][split])
                atomic_json(output/'report.json', report)
                for name, summary in report['splits'][split].items():
                    pair = summary['paired_with_clean']
                    log.info('%s %s n=%d accuracy=%.4f recall_fake=%s FPR=%s flips=%d '
                             'clean_minus_variant=%.4f paired_CI=%s', split, name, summary['n'],
                             summary['accuracy'], summary['recall_fake'], summary['false_positive_rate'],
                             pair['changed_decisions'], pair['clean_minus_variant_accuracy'], pair['paired_group_bootstrap_95ci'])
        except BaseException:
            log.exception('Motion audit failed/interrupted; no training step was performed')
            raise
        finally:
            for name, expected in hashes.items():
                if file_sha(run/name) != expected:
                    raise RuntimeError(f'Training artifact unexpectedly changed during audit: {name}')
        report['training_artifacts_unchanged'] = True
        atomic_json(output/'report.json', report)
        log.info('MOTION AUDIT COMPLETE %s', output)


if __name__ == '__main__':
    main()
