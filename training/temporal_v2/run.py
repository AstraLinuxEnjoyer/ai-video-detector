"""One command for preparation, training, monitoring and held-out evaluation."""
import argparse
import json
import platform
import shutil
import sys
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
from .dashboard import Monitor, start_server
from .engine import train, evaluate, predictions
from .media import prepare
from .flow import FLOW_PARAMS, FLOW_VERSION, audit_headers, is_motion, quarantine_labels, rotation_for

DEFAULTS = dict(epochs=12, videos_per_epoch=1000, train_clips=2, val_clips=3,
                val_videos=800, test_videos=800, batch_size=4, image_size=112, decode_size=128,
                clip_length=16, fps=16, sequence_fps=0., seed=42, workers=2, decode_workers=2,
                backbone_lr=1e-5, head_lr=1e-4, cache_gb=12., reserve_gb=10., patience=5,
                min_source_count=10, log_every=20, save_every=20, preview_count=4, device='cuda',
                input_mode='rgb', flow_scale=.5, max_fake_source_fraction=.6)

MOTION_DEFAULTS = dict(input_mode='flow', fps=4, clip_length=9, videos_per_epoch=500)


def choose_validation(records, args):
    # Matching eligibility is frozen; validation prioritizes source coverage rather
    # than reproducing the generator dominance of rare paired spatial/FPS cells.
    return fixed_selection(records, args.val_videos, args.seed)


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
        for name in ('initial.pt', 'dimensions.json'):
            if file_sha(run/name) != config[name+'_sha256']:
                raise ValueError(f'Frozen artifact changed: {name}')
    else:
        source = (source or ROOT/'cache/artifact_spatial/runs/hybrid_v2').resolve()
        if run == source:
            raise ValueError('Temporal and spatial run directories must differ')
        config = {**DEFAULTS, **(MOTION_DEFAULTS if given.get('input_mode') == 'flow' else {}), **given}
        for k in DEFAULTS:
            if k not in ('device', 'input_mode', 'sequence_fps', 'workers') and config[k] <= 0:
                raise ValueError(f'{k} must be positive')
        if config['device'] not in ('cuda', 'cpu') or config['workers'] < 0 or config['sequence_fps'] < 0:
            raise ValueError('Invalid device/workers/sequence FPS')
        if config['input_mode'] not in ('rgb', 'flow'):
            raise ValueError('Input mode must be rgb or flow')
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
        log.info('Initialize R(2+1)D-18 from Kinetics-400, NOT from spatial/old temporal weights')
        model = build_temporal_model(True, 'flow') if config['input_mode'] == 'flow' else build_temporal_model(True)
        atomic_torch(run/'initial.pt', model.state_dict())
        config.update(source_run=str(source), recipe='temporal-motion-v1' if config['input_mode'] == 'flow' else 'temporal-v2',
                      **{name+'_sha256': file_sha(run/name) for name in ('initial.pt', 'dimensions.json')})
        if config['input_mode'] == 'flow':
            import cv2
            config.update(flow_version=FLOW_VERSION, opencv_version=cv2.__version__, flow_parameters=FLOW_PARAMS)
        # Config marks successful initialization; no first-model artifact is modified.
        atomic_json(run/'config.json', config)
    assert_disjoint(payload['splits'])
    args = argparse.Namespace(**config)
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
    preview = rotation.take(args.videos_per_epoch)
    audit = dict(original_counts={k: len(v) for k, v in payload['splits'].items()},
                 eligible_counts={k: len(v) for k, v in eligible.items()},
                 excluded_sequences={k: sum(r.get('kind') == 'image_sequence' and args.sequence_fps <= 0 for r in v)
                                     for k, v in payload['splits'].items()},
                 excluded_bad_train_headers=bad_train_headers+len(eligible['train'])-len(training),
                 matching=rotation.audit, first_epoch_videos=len(preview),
                 split_hash=payload['fingerprint'], sequence_fps=args.sequence_fps,
                 caveat='No new data downloaded. Known origin groups and exact hashes, not exhaustive near-duplicate detection.')
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
    parser.add_argument('--source-run', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--input', type=Path, default=ROOT/'test_videos')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-dashboard', action='store_true')
    for name, default in DEFAULTS.items():
        parser.add_argument('--'+name.replace('_', '-'), type=type(default), default=argparse.SUPPRESS)
    parsed = parser.parse_args(argv)
    given = {k: v for k, v in vars(parsed).items() if k in DEFAULTS}
    if parsed.motion:
        if given.get('input_mode', 'flow') != 'flow':
            parser.error('--motion conflicts with --input-mode rgb')
        given['input_mode'] = 'flow'
    motion = given.get('input_mode') == 'flow'
    default_run = ROOT/('cache/temporal_v2/runs/motion_flow_v1' if motion else 'cache/temporal_v2/runs/r2plus1d_v1')
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
            monitor.publish(config=vars(args), data=json.loads((run/'data_audit.json').read_text()))
            if parsed.prepare_only:
                cache = FrameCache(run, args.cache_gb, args.reserve_gb)
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
