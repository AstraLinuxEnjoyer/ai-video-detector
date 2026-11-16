"""One command for preparation, training, monitoring and held-out evaluation."""
import argparse
import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from filelock import FileLock

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.data import assert_disjoint, fixed_selection, EXTENSIONS
from training.spatial.engine import atomic_torch, seed_all
from training.spatial.lineage import file_sha
from training.spatial.media import FrameCache
from training.temporal_model import build_temporal_model
from training.artifact_spatial.metadata import attach_dimensions
from training.artifact_spatial.sampling import MatchedRotation
from training.final_cohort import MOTION_FINAL, MOTION_PARENT, SPATIAL_FINAL, heldout_selection
from training.final_metrics import comparison, operating_point, read_json
from .dashboard import Monitor, start_server
from .engine import train, evaluate, predictions
from .media import prepare
from .flow import FLOW_PARAMS, FLOW_VERSION, audit_headers, is_motion, quarantine_labels, rotation_for
from .paired_batch import build_plans, verify_plans

DEFAULTS = dict(epochs=12, videos_per_epoch=1000, train_clips=2, val_clips=3,
                val_videos=800, test_videos=800, batch_size=4, image_size=112, decode_size=128,
                clip_length=16, fps=16, sequence_fps=0., seed=42, workers=2, decode_workers=2,
                backbone_lr=1e-5, head_lr=1e-4, cache_gb=12., reserve_gb=10., patience=5,
                min_source_count=10, log_every=20, save_every=20, preview_count=4, device='cuda',
                input_mode='rgb', flow_scale=.5, max_fake_source_fraction=.6, frame_view='crop',
                final_tune=False, coverage_tune=False, paired_batch=False)

MOTION_DEFAULTS = dict(input_mode='flow', fps=4, clip_length=9, videos_per_epoch=500)
COVERAGE_RECIPE = 'temporal-motion-coverage-v1'
PAIRED_RECIPE = 'temporal-motion-paired-v1'
PREVIOUS_ATTEMPTS_HASH = 'previous_attempts_sha256'


def verify_frozen_artifacts(run, config):
    artifacts = {'initial.pt': 'initial.pt_sha256', 'dimensions.json': 'dimensions.json_sha256'}
    if config.get('paired_batch'):
        artifacts['previous_attempts.json'] = PREVIOUS_ATTEMPTS_HASH
    for name, key in artifacts.items():
        if key not in config:
            raise ValueError(f'Frozen artifact checksum missing: {name}')
        if file_sha(run/name) != config[key]:
            raise ValueError(f'Frozen artifact changed: {name}')


def initialize_from_motion(run, source, config, payload, log):
    coverage = config['coverage_tune']
    paired = config['paired_batch']
    parent_run = (MOTION_FINAL.with_name('motion_coverage_v1') if paired else
                  MOTION_FINAL if coverage else MOTION_PARENT)
    parent = read_json(parent_run/'config.json')
    best = read_json(parent_run/'best.json')
    expected_split = (payload['fingerprint'] if coverage else
                      read_json(source/'final_cohort.json')['parent_fingerprint'])
    expected_recipe = (COVERAGE_RECIPE if paired else
                       'temporal-final-motion-v1' if coverage else 'temporal-motion-v1')
    if (parent['recipe'] != expected_recipe or best['split_hash'] != expected_split or
            any(parent[k] != config[k] for k in ('decode_size','image_size','fps','clip_length','flow_scale'))):
        raise ValueError('Motion parent weights and audited split/input settings do not match')
    if coverage:
        import cv2
        if (parent['frame_view'] != config['frame_view'] or parent['flow_version'] != FLOW_VERSION
                or parent['flow_parameters'] != FLOW_PARAMS or parent['opencv_version'] != cv2.__version__):
            raise ValueError('Coverage continuation requires the same full-frame flow preprocessing')
        if not (parent_run/'final_report.json').is_file():
            raise ValueError('Finish the previous final motion run before starting coverage tuning')
    weights = parent_run/'best.pt'
    sha = file_sha(weights)
    atomic_torch(run/'initial.pt', torch.load(weights, map_location='cpu', weights_only=True))
    if file_sha(weights) != sha or read_json(parent_run/'best.json') != best:
        raise RuntimeError('Motion parent changed during initialization')
    seen = set(read_json(parent_run/'parent_best.json')['seen_ids']) if coverage else set()
    for index in range(1, best['epoch']+1):
        seen.update(read_json(parent_run/f'epoch_{index:03}_plan.json')['ids'])
    atomic_json(run/'parent_best.json', dict(best, weights_sha256=sha, seen_ids=sorted(seen)))
    if paired:
        tried = set(read_json(parent_run/'coverage.json')['ids']) | seen
        atomic_json(run/'previous_attempts.json', dict(ids=sorted(tried)))
        config[PREVIOUS_ATTEMPTS_HASH] = file_sha(run/'previous_attempts.json')
    headers = parent_run/'motion_headers.jsonl'
    if headers.exists() and not (run/'motion_headers.jsonl').exists():
        shutil.copyfile(headers, run/'motion_headers.jsonl')
    log.info('Initialize from %s best epoch %d; %d inherited videos; full-frame geometry, new optimizer',
             parent_run.name, best['epoch'], len(seen))


def project_coverage(rotation, preview, args):
    seen = set(args.previously_seen)
    epochs = []
    for epoch in range(1, args.epochs+1):
        plan = preview if epoch == 1 else rotation.take(args.videos_per_epoch)
        novel = [r for r in plan if r['id'] not in seen]
        epochs.append(dict(epoch=epoch, videos=len(plan), new=len(novel),
                           new_by_label=dict(Counter(r['label'] for r in novel))))
        seen.update(r['id'] for r in plan)
    return dict(epochs=epochs, inherited=len(args.previously_seen),
                novel_total=len(seen)-len(args.previously_seen),
                caveat='Scheduled upper bound, not training telemetry; early stopping may shorten the run.')


def choose_validation(records, args):
    # Matching eligibility is frozen; validation prioritizes source coverage rather
    # than reproducing the generator dominance of rare paired spatial/FPS cells.
    select = heldout_selection if getattr(args, 'final_tune', False) else fixed_selection
    return select(records, args.val_videos, args.seed)


def initialize(run, source, given, log, publish=None):
    if (run/'config.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config.get('input_mode') == 'flow' and 'max_fake_source_fraction' not in config:
            if (run/'checkpoint.pt').exists():
                raise ValueError('Motion sampling policy missing from trained run; use a NEW run directory')
            # Upgrade only the prepared, never-trained motion experiment.
            config.update(videos_per_epoch=500, max_fake_source_fraction=.6, flow_parameters=FLOW_PARAMS)
            atomic_json(run/'config.json', config)
        if any(config.get(k, DEFAULTS.get(k)) != v for k, v in given.items()):
            raise ValueError('Run settings are frozen. Use a NEW --run-dir for different settings')
        if source and str(source.resolve()) != config['source_run']:
            raise ValueError('Cannot change split source in an existing run')
        payload = json.loads((run/'splits.json').read_text(encoding='utf-8'))
        if fingerprint(payload['splits']) != payload['fingerprint']:
            raise ValueError('Frozen split fingerprint mismatch')
        verify_frozen_artifacts(run, config)
    else:
        source = (source or (SPATIAL_FINAL if given.get('coverage_tune') else
                             ROOT/'cache/artifact_spatial/runs/hybrid_v2')).resolve()
        if run == source:
            raise ValueError('Temporal and spatial run directories must differ')
        config = {**DEFAULTS, **(MOTION_DEFAULTS if given.get('input_mode') == 'flow' else {}), **given}
        for k in DEFAULTS:
            if k not in ('device', 'input_mode', 'sequence_fps', 'workers', 'frame_view',
                         'final_tune', 'coverage_tune', 'paired_batch') and config[k] <= 0:
                raise ValueError(f'{k} must be positive')
        if config['device'] not in ('cuda', 'cpu') or config['workers'] < 0 or config['sequence_fps'] < 0:
            raise ValueError('Invalid device/workers/sequence FPS')
        if config['input_mode'] not in ('rgb', 'flow'):
            raise ValueError('Input mode must be rgb or flow')
        if config['frame_view'] not in ('crop', 'full') or (
                config['frame_view'] == 'full' and config['input_mode'] != 'flow'):
            raise ValueError('Full-frame mode requires optical flow')
        if config['final_tune'] and (config['frame_view'] != 'full' or config['input_mode'] != 'flow'):
            raise ValueError('Final tuning requires full-frame flow')
        if config['coverage_tune'] and not config['final_tune']:
            raise ValueError('Coverage tuning requires final tuning')
        if config['paired_batch'] and not config['coverage_tune']:
            raise ValueError('Paired batches require coverage tuning')
        if not 0 < config['max_fake_source_fraction'] <= 1:
            raise ValueError('Source fraction must be in (0, 1]')
        if config['input_mode'] == 'flow' and config['sequence_fps']:
            raise ValueError('Motion v1 excludes image sequences without original timing metadata')
        if config['image_size'] > config['decode_size'] or config['image_size'] < 32 or config['clip_length'] < 4:
            raise ValueError('Invalid clip dimensions')
        if config['videos_per_epoch'] % 2 or min(config['val_videos'], config['test_videos']) < 2:
            raise ValueError('Need even train count and at least two validation/test videos')
        if config['input_mode'] == 'flow' and config['val_videos'] % 2:
            raise ValueError('Motion validation needs an even class-balanced count')
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise RuntimeError('ffmpeg and ffprobe must be on PATH')
        if shutil.disk_usage(run).free < (config['reserve_gb']+2)*1024**3:
            raise OSError('Insufficient free disk space for checkpoints and reserve')
        payload = json.loads((source/'splits.json').read_text(encoding='utf-8'))
        if fingerprint(payload['splits']) != payload['fingerprint']:
            raise ValueError('Source split snapshot is corrupt')
        assert_disjoint(payload['splits'])
        dimensions = json.loads((source/'dimensions.json').read_text())
        atomic_json(run/'splits.json', payload)
        atomic_json(run/'dimensions.json', dimensions)
        seed_all(config['seed'])
        if config['final_tune']:
            initialize_from_motion(run, source, config, payload, log)
        else:
            log.info('Initialize R(2+1)D-18 from Kinetics-400, NOT from spatial/old temporal weights')
            model = build_temporal_model(True, 'flow') if config['input_mode'] == 'flow' else build_temporal_model(True)
            atomic_torch(run/'initial.pt', model.state_dict())
        recipe = (PAIRED_RECIPE if config['paired_batch'] else
                  COVERAGE_RECIPE if config['coverage_tune'] else
                  'temporal-final-motion-v1' if config['final_tune'] else
                  'temporal-motion-full-frame-v2' if config['frame_view'] == 'full' else
                  'temporal-motion-v1' if config['input_mode'] == 'flow' else 'temporal-v2')
        config.update(source_run=str(source), recipe=recipe,
                      **{name+'_sha256': file_sha(run/name) for name in ('initial.pt', 'dimensions.json')})
        if config['input_mode'] == 'flow':
            import cv2
            config.update(flow_version=FLOW_VERSION, opencv_version=cv2.__version__, flow_parameters=FLOW_PARAMS)
        # Config marks successful initialization; no first-model artifact is modified.
        atomic_json(run/'config.json', config)
    assert_disjoint(payload['splits'])
    args = argparse.Namespace(**config)
    if getattr(args, 'final_tune', False):
        args.previously_seen = read_json(run/'parent_best.json')['seen_ids']
        if not set(args.previously_seen) <= {row['id'] for row in payload['splits']['train']} | set(
                read_json(Path(config['source_run'])/'final_cohort.json')['excluded_train_decode_ids']):
            raise ValueError('Motion parent exposure history conflicts with final training splits')
    dimensions = json.loads((run/'dimensions.json').read_text())
    eligible = {key: [r for r in rows if args.sequence_fps > 0 or r.get('kind') != 'image_sequence']
                for key, rows in payload['splits'].items()}
    bad_train_headers = 0
    if is_motion(args):
        import cv2
        if config.get('flow_version', FLOW_VERSION) != FLOW_VERSION:
            raise ValueError('Motion preprocessing changed; use a NEW run directory')
        if config.get('opencv_version') != cv2.__version__:
            raise ValueError('Optical-flow library version changed; use a NEW run directory')
        if config.get('flow_parameters', FLOW_PARAMS) != FLOW_PARAMS:
            raise ValueError('Optical-flow parameters changed; use a NEW run directory')
        eligible = quarantine_labels(eligible, run, log)
        attached = attach_dimensions(eligible['train'], dimensions)
        bad_train_headers = len(eligible['train'])-len(attached)
        eligible['train'] = attached
        eligible = audit_headers(eligible, run, args, log, publish)
        args.motion_cohort_hash = json.loads((run/'motion_cohort.json').read_text())['fingerprint']
        log.info('MOTION INPUT 2 velocity channels; %d RGB frames -> %d flow fields at %d FPS; '
                 'no RGB to classifier; BN statistics learned on flow; flow_scale=%.3f',
                 args.clip_length, args.clip_length-1, args.fps, args.flow_scale)
    training = attach_dimensions(eligible['train'], dimensions)
    rotation = rotation_for(training, args)
    paired_summary = {}
    if getattr(args, 'paired_batch', False):
        plan_file = run/'paired_plan.json'
        ids = {row['id']: row for row in training}
        if 'paired_plan_hash' not in config:
            tried = set(read_json(run/'previous_attempts.json')['ids'])
            if not tried <= {row['id'] for row in payload['splits']['train']} | set(read_json(
                    Path(config['source_run'])/'final_cohort.json')['excluded_train_decode_ids']):
                raise ValueError('Previous attempts conflict with the train split')
            plans = build_plans(training, tried, args)
            atomic_json(plan_file, plans)
            config['paired_plan_hash'] = plans['fingerprint']
            atomic_json(run/'config.json', config)
        plans = read_json(plan_file)
        args.paired_plan_hash = config['paired_plan_hash']
        verify_plans(plans, args.paired_plan_hash, ids.keys(), args.epochs, args.videos_per_epoch)
        preview = [ids[i] for i in plans['plans'][0]['ids']]
        paired_summary = dict(new_total=plans['new_total'], eligible_DFDC_groups=plans['paired_groups'],
                              epochs=[{k: p[k] for k in ('epoch','sources','new_by_label')}
                                      for p in plans['plans']])
        log.info('PAIRED PLAN epochs=%d videos/epoch=%d fresh DFDC pairs/epoch=%d total new=%d',
                 args.epochs, args.videos_per_epoch, args.videos_per_epoch//4, plans['new_total'])
    else:
        preview = rotation.take(args.videos_per_epoch)
    if getattr(args, 'coverage_tune', False) and not paired_summary and not (run/'coverage_plan.json').exists():
        projected = project_coverage(rotation, preview, args)
        atomic_json(run/'coverage_plan.json', projected)
        log.info('COVERAGE PLAN novel=%d inherited=%d eligible=%d per_epoch=%s',
                 projected['novel_total'], projected['inherited'], len(training),
                 [row['new'] for row in projected['epochs']])
    audit = dict(original_counts={k: len(v) for k, v in payload['splits'].items()},
                 eligible_counts={k: len(v) for k, v in eligible.items()},
                 excluded_sequences={k: sum(r.get('kind') == 'image_sequence' and args.sequence_fps <= 0 for r in v)
                                     for k, v in payload['splits'].items()},
                 excluded_bad_train_headers=bad_train_headers+len(eligible['train'])-len(training),
                 matching=rotation.audit, first_epoch_videos=len(preview),
                 split_hash=payload['fingerprint'], sequence_fps=args.sequence_fps,
                 caveat='No new data downloaded. Known origin groups and exact hashes, not exhaustive near-duplicate detection.')
    if paired_summary:
        audit['paired_batch'] = paired_summary
    atomic_json(run/'data_audit.json', audit)
    log.info('SPLITS %s excluded_sequences=%s bad_train_headers=%d', audit['eligible_counts'],
             audit['excluded_sequences'], audit['excluded_bad_train_headers'])
    return args, payload, training, eligible


def evaluation_coverage(requested, rows, unsupported):
    expected = {r['id'] for r in requested}
    evaluated = {r['id'] for r in rows}
    rejected = {r['id'] for r in unsupported}
    if (len(expected) != len(requested) or len(evaluated) != len(rows)
            or len(rejected) != len(unsupported) or evaluated & rejected
            or evaluated | rejected != expected):
        raise ValueError('Evaluation coverage does not account for every requested ID exactly once')

    def counts(selected):
        ids = {r['id'] for r in selected}
        predictions = [r for r in rows if r['id'] in ids]
        correct = sum((r['score'] >= .5) == bool(r['label']) for r in predictions)
        missing = len(ids)-len(predictions)
        return dict(requested=len(ids), evaluated=len(predictions), unsupported=missing,
                    coverage=len(predictions)/len(ids), correct_all_requested=correct,
                    accuracy_all_requested_bounds=[correct/len(ids), (correct+missing)/len(ids)])

    return dict(**counts(requested), unsupported_videos=unsupported,
                by_source={source: counts([r for r in requested if r['source'] == source])
                           for source in sorted({r['source'] for r in requested})},
                by_class={str(label): counts([r for r in requested if r['label'] == label])
                          for label in sorted({r['label'] for r in requested})},
                caveat='Accuracy/recall/FPR above describe evaluated videos only. Bounds count all '
                       'requested videos with abstentions as incorrect/correct; they are not confidence intervals.')


def heldout_evaluation(model, records, cache, tag, args, log, run, monitor):
    prepared = prepare(records, cache, tag, args.val_clips, args, log, run,
                       monitor.publish, allow_short=True)
    unsupported = json.loads((run/f'{tag}_decode.json').read_text())['unsupported']
    if prepared:
        rows, report = evaluate(model, prepared, args, torch.device(args.device), log, monitor, tag)
    else:
        rows, report = [], dict(n=0, accuracy=None, by_source={})
    report.update(status='complete_with_unsupported' if unsupported else 'complete',
                  metric_scope='evaluated_videos_only',
                  input_contract=dict(clip_length=args.clip_length, fps=args.fps,
                                      input_mode=getattr(args, 'input_mode', 'rgb'),
                                      minimum_duration=args.clip_length/args.fps),
                  coverage=evaluation_coverage(records, rows, unsupported))
    coverage = report['coverage']
    log.info('%s COVERAGE requested=%d evaluated=%d unsupported=%d coverage=%.4f '
             'accuracy_all_requested_bounds=%s', tag, coverage['requested'], coverage['evaluated'],
             coverage['unsupported'], coverage['coverage'], coverage['accuracy_all_requested_bounds'])
    return rows, report


def final_test(run, splits, args, log, monitor):
    if (run/'final_report.json').exists():
        monitor.publish(final=json.loads((run/'final_report.json').read_text()))
        return
    device = torch.device(args.device)
    model = (build_temporal_model(False, 'flow') if is_motion(args) else build_temporal_model(False)).to(device)
    model.load_state_dict(torch.load(run/'best.pt', map_location='cpu', weights_only=True))
    cache = FrameCache(run, args.cache_gb, args.reserve_gb)
    reports = dict(caveat='Tests were inspected during spatial development, not a fresh blind benchmark. '
                   'OOD can be fake-only. Temporal and spatial exclusions may differ; compare common IDs.')
    if getattr(args, 'final_tune', False):
        return final_motion_test(run, splits, args, log, monitor, model, cache, reports)
    for split in ('test', 'ood'):
        records = fixed_selection(splits[split], args.test_videos, args.seed)
        if not records:
            reports[split] = dict(status='empty')
            continue
        rows, reports[split] = heldout_evaluation(model, records, cache, split, args, log, run, monitor)
        predictions(run/f'{split}_predictions.csv', rows)
        atomic_json(run/f'{split}_report.json', reports[split])
    atomic_json(run/'final_report.json', reports)
    monitor.publish(final=reports)


def final_motion_test(run, splits, args, log, monitor, model, cache, reports):
    baseline = build_temporal_model(False, 'flow').to(args.device)
    baseline.load_state_dict(torch.load(run/'initial.pt', map_location='cpu', weights_only=True))
    parent_args = (args if getattr(args, 'coverage_tune', False) else
                   argparse.Namespace(**dict(vars(args), frame_view='crop')))
    validation = choose_validation(splits['val'], args)
    baseline_val = run/'parent_operating_point.json'
    if not baseline_val.exists():
        prefix = run/'baseline'
        prefix.mkdir(exist_ok=True)
        prepared = prepare(validation, FrameCache(prefix, args.cache_gb/2, args.reserve_gb), 'val',
                           args.val_clips, parent_args, log, prefix, monitor.publish)
        clean, stress = (evaluate(baseline, prepared, parent_args, torch.device(args.device), log, monitor,
                                  f'parent_val/{mode}', stress=mode=='stress')[0] for mode in ('clean','stress'))
        atomic_json(baseline_val, operating_point(clean, stress))
    thresholds = dict(baseline=read_json(baseline_val)['threshold'], candidate=read_json(run/'best.json')['threshold'])
    for split in ('test', 'ood'):
        records = heldout_selection(splits[split], args.test_videos, args.seed)
        if not records:
            continue
        sides = {mode: [] for mode in ('clean', 'stress')}
        for name, net, config in (('baseline', baseline, parent_args), ('candidate', model, args)):
            prefix = run/name if name == 'baseline' else run
            source_cache = FrameCache(prefix, args.cache_gb/2, args.reserve_gb) if name == 'baseline' else cache
            prepared = prepare(records, source_cache, split, args.val_clips, config,
                               log, prefix, monitor.publish, allow_short=True)
            for mode in sides:
                rows = (evaluate(net, prepared, config, torch.device(args.device), log, monitor,
                                 f'{split}/{mode}/{name}', stress=mode=='stress')[0] if prepared else [])
                predictions(run/f'{split}_{name}_{mode}.csv', rows)
                sides[mode].append(rows)
        reports[split] = comparison(run, split, sides, thresholds, records)
    atomic_json(run/'final_report.json', reports)
    monitor.publish(final=reports)


def external(run, path, payload, args, log, monitor):
    train_ids = {r['id'] for r in payload['splits']['train']}
    records, seen = [], {}
    for label, name in enumerate(('real', 'fake')):
        for video in sorted((path/name).rglob('*')):
            if not video.is_file() or video.suffix.lower() not in EXTENSIONS:
                continue
            sha = file_sha(video)
            if sha in train_ids:
                raise ValueError(f'External video matches a training file: {video}')
            if sha in seen:
                if seen[sha] != label:
                    raise ValueError('Conflicting labels on duplicate external files')
                continue
            seen[sha] = label
            stat = video.stat()
            records.append(dict(id=sha, group=sha, path=str(video.resolve()), label=label,
                                source=name, size=stat.st_size, mtime=stat.st_mtime_ns))
    if not records:
        raise ValueError('No videos in input/real and input/fake')
    from datetime import datetime
    output = run/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True)
    model = (build_temporal_model(False, 'flow') if is_motion(args) else build_temporal_model(False)).to(args.device)
    model.load_state_dict(torch.load(run/'best.pt', map_location='cpu', weights_only=True))
    rows, report = heldout_evaluation(model, records, FrameCache(output, args.cache_gb, args.reserve_gb),
                                     'external', args, log, output, monitor)
    predictions(output/'predictions.csv', rows)
    atomic_json(output/'report.json', report)
    monitor.publish(external=report)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--motion', action='store_true', help='Motion-only experiment; never resume RGB weights')
    parser.add_argument('--final-tune', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--coverage-tune', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--paired-batch', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--source-run', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--evaluate-only', action='store_true')
    modes.add_argument('--audit-only', action='store_true', help='Verify the first epoch plan without decoding clips')
    parser.add_argument('--input', type=Path, default=ROOT/'test_videos')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-dashboard', action='store_true')
    for name, default in DEFAULTS.items():
        if name in ('final_tune', 'coverage_tune', 'paired_batch'):
            continue
        parser.add_argument('--'+name.replace('_', '-'), type=type(default), default=argparse.SUPPRESS)
    parsed = parser.parse_args(argv)
    given = {k: v for k, v in vars(parsed).items() if k in DEFAULTS}
    if parsed.motion:
        if given.get('input_mode', 'flow') != 'flow':
            parser.error('--motion conflicts with --input-mode rgb')
        given['input_mode'] = 'flow'
    motion = given.get('input_mode') == 'flow'
    if given.get('frame_view') == 'full' and not motion and parsed.run_dir is None:
        parser.error('--frame-view full requires --motion')
    run_name = ('motion_paired_dfdc_v1' if given.get('paired_batch') else
                'motion_coverage_v1' if given.get('coverage_tune') else
                'final_motion_v1' if given.get('final_tune') else 'motion_full_frame_v2' if given.get('frame_view') == 'full' else
                'motion_flow_v1' if motion else 'r2plus1d_v1')
    default_run = ROOT/'cache/temporal_v2/runs'/run_name
    run = (parsed.run_dir or default_run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock_root = ROOT/'cache/spatial_v2'
    lock_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_root/'training.lock'), timeout=0), FileLock(str(run/'run.lock'), timeout=0):
        log = logger_for(run)
        monitor = Monitor(run)
        server = None
        try:
            monitor.publish(status='preparing', phase='initialize', error=None)
            if not parsed.no_dashboard:
                server, url = start_server(run, parsed.port)
                log.info('DASHBOARD %s', url)
                monitor.publish(url=url)
            args, payload, training, eligible = initialize(run, parsed.source_run, given, log, monitor.publish)
            atomic_json(run/'environment.json', dict(python=sys.version, executable=sys.executable,
                        platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
            monitor.publish(config={key: value for key, value in vars(args).items() if key != 'previously_seen'},
                            data=json.loads((run/'data_audit.json').read_text()))
            if parsed.audit_only:
                monitor.publish(status='prepared', phase='ready')
                log.info('AUDITED. Run without --audit-only to train.')
                return
            if parsed.prepare_only:
                cache = FrameCache(run, args.cache_gb, args.reserve_gb)
                if getattr(args, 'paired_batch', False):
                    ids = {r['id']: r for r in training}
                    first = [ids[i] for i in read_json(run/'paired_plan.json')['plans'][0]['ids']]
                else:
                    first = rotation_for(training, args).take(args.videos_per_epoch)
                validation = choose_validation(eligible['val'], args)
                prepare(first, cache, 'train_0', args.train_clips, args, log, run, monitor.publish)
                prepare(validation, cache, 'val', args.val_clips, args, log, run, monitor.publish)
                monitor.publish(status='prepared', phase='ready')
                log.info('PREPARED. Run the same command without --prepare-only to train.')
                return
            if parsed.evaluate_only:
                external(run, parsed.input, payload, args, log, monitor)
            else:
                validation = choose_validation(eligible['val'], args)
                model, _ = train(run, payload, training, validation, args, log, monitor)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                final_test(run, eligible, args, log, monitor)
            monitor.publish(status='complete', phase='finished')
            log.info('COMPLETE. Best weights: %s', run/'best.pt')
        except KeyboardInterrupt:
            monitor.publish(status='interrupted', phase='stopped')
            log.warning('Interrupted. Restart the same command; resumes last durable checkpoint, up to save-every batches replayed.')
        except Exception as exc:
            monitor.publish(status='failed', error=str(exc))
            log.exception('Run failed. Last durable checkpoint retained; no samples replaced.')
            raise
        finally:
            monitor.close()
            if server:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    main()
