"""Motion-only input. RGB is used by the estimator, never by the classifier."""
import io
import json
import math
import random
from collections import Counter
from concurrent.futures import as_completed
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from tqdm import tqdm

from training.artifact_spatial.sampling import MatchedRotation, spatial_bucket
from training.spatial.common import atomic_json, event, fingerprint
from training.spatial.media import SourceChangedError, decoding_pool, probe

FLOW_VERSION = 'farneback-residual-velocity-v1'
LABEL_POLICY = 'quarantine-figure-physical-robot-demo-v1'
FLOW_PARAMS = dict(pyr_scale=.5, levels=4, winsize=21, iterations=5, poly_n=7,
                   poly_sigma=1.5, flags=0)


def is_motion(args):
    return getattr(args, 'input_mode', 'rgb') == 'flow'


def quarantine_labels(splits, run, log):
    suspect = 'figure status update - openai speech-to-speech reasoning.mp4'
    filtered, rejected = {}, []
    for split, rows in splits.items():
        filtered[split] = []
        for row in rows:
            if row['label'] == 1 and Path(row['path']).name.lower() == suspect:
                rejected.append(dict(id=row['id'], path=row['path'], source=row['source'],
                                     split=split, supplied_label=1,
                                     reason='Physical robot reasoning demo is not verified generated imagery; '
                                            'AI control/speech does not label camera pixels fake'))
            else:
                filtered[split].append(row)
    atomic_json(run/'label_quarantine.json', dict(policy=LABEL_POLICY, records=rejected,
                caveat='No original changed or automatic real relabeling; known suspect omitted from this motion experiment'))
    if rejected:
        log.warning('LABEL QUARANTINE %d known suspect videos; see label_quarantine.json. Originals unchanged.', len(rejected))
    return filtered


class MotionInputError(ValueError):
    """Valid video outside the motion sampling contract."""


def check_metadata(meta, args):
    fps = meta.get('fps', 0)
    if not math.isfinite(fps) or fps + 1e-3 < args.fps:
        raise MotionInputError(f"Native FPS {fps} < motion FPS {args.fps}; no upsampling allowed")


def motion_bucket(record):
    fps = record['metadata']['fps']
    band = '4-8' if fps < 8 else '8-16' if fps < 16 else '16-24' if fps < 24 else '24-32' if fps < 32 else '32+'
    return spatial_bucket(record) + '/fps_' + band


def rotation_for(records, args, state=None):
    kwargs = dict(bucket=motion_bucket, max_source_fraction_by_class={1: getattr(args, 'max_fake_source_fraction', .6)}) if is_motion(args) else {}
    rotation = MatchedRotation(records, args.seed, state=state, **kwargs)
    if is_motion(args):
        rotation.audit['max_fake_source_fraction'] = getattr(args, 'max_fake_source_fraction', .6)
        rotation.audit['caveat'] = 'Coarse resolution/aspect/native-FPS matching, not exact codec or semantic matching'
    return rotation


def audit_headers(splits, run, args, log, publish=None):
    """Resume cheap header inspection; freeze exclusions before selecting any clips."""
    journal = run/'motion_headers.jsonl'
    index = {}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                index[row['id']] = row
            except json.JSONDecodeError:
                continue
    records = [r for key in ('train', 'val') for r in splits[key]]
    pending = [r for r in records if r['id'] not in index]
    log.info('MOTION HEADERS remaining=%d cached=%d; originals unchanged', len(pending), len(index))

    def one(record):
        path = Path(record['path'])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime']):
            raise SourceChangedError(f'Frozen source changed: {path}')
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or 0)
                duration = float(stream.duration*stream.time_base) if stream.duration is not None else float(container.duration or 0)/av.time_base
                meta = dict(width=stream.codec_context.width, height=stream.codec_context.height,
                            fps=fps, duration=duration)
            if not all(math.isfinite(meta[k]) and meta[k] > 0 for k in meta):
                raise ValueError('Incomplete header')
        except (OSError, ValueError, IndexError):
            meta = probe(path)
        return dict(id=record['id'], size=record['size'], mtime=record['mtime'], metadata=meta)

    with decoding_pool(args.decode_workers) as pool:
        futures = {pool.submit(one, r): r for r in pending}
        for step, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc='MOTION HEADERS'), 1):
            row = future.result()  # Broken or changed sources are fatal, not silently replaced.
            index[row['id']] = row
            event(journal, row)
            if step % 1000 == 0 or step == len(pending):
                log.info('MOTION HEADERS %d/%d', step, len(pending))
                if publish:
                    publish(phase='motion_headers', done=step, total=len(pending))
    result = dict(splits)
    exclusions, matching = {}, {}
    for key in ('train', 'val'):
        accepted, rejected = [], []
        for record in splits[key]:
            saved = index[record['id']]
            if (saved['size'], saved['mtime']) != (record['size'], record['mtime']):
                raise SourceChangedError('Motion header snapshot mismatch')
            meta = saved['metadata']
            reason = None
            try:
                check_metadata(meta, args)
            except MotionInputError:
                reason = 'native_fps_below_target'
            if meta['duration'] < args.clip_length/args.fps:
                reason = 'clip_too_short'
            if reason:
                rejected.append(dict(id=record['id'], source=record['source'], label=record['label'],
                                     reason=reason, metadata=meta))
            else:
                accepted.append(dict(record, metadata=meta))
        sampler = rotation_for(accepted, args)
        cells = set(sampler.cells)
        unmatched = [r for r in accepted if motion_bucket(r) not in cells]
        result[key] = [r for r in accepted if motion_bucket(r) in cells]
        exclusions[key] = dict(input_contract=rejected,
                               unmatched=[dict(id=r['id'], source=r['source'], label=r['label'],
                                               reason='no_other_class_in_fps_spatial_cell') for r in unmatched])
        matching[key] = sampler.audit
        log.info('MOTION ELIGIBILITY %s original=%d eligible=%d contract_excluded=%d unmatched=%d sources=%s',
                 key, len(splits[key]), len(result[key]), len(rejected), len(unmatched),
                 dict(Counter(r['source'] for r in result[key])))
    report = dict(version=FLOW_VERSION, opencv=cv2.__version__, parameters=FLOW_PARAMS,
                  input='2 channels: residual vx/vy, fixed normalized frame-width units per second',
                  fps=args.fps, decoded_frames=args.clip_length, motion_steps=args.clip_length-1,
                  flow_scale=args.flow_scale, camera_compensation='per-pair median translation only',
                  eligible_counts={k: len(v) for k, v in result.items()}, exclusions=exclusions, matching=matching,
                  caveat='Average native FPS is a header check, not a guarantee of constant timestamps. '
                         'Static/held source frames can remain. Low-texture flow, cuts, blur, rotation and parallax '
                         'can create estimator errors. No physics or object annotations are available.')
    cohort = {key: sorted((r['id'], r['metadata'], motion_bucket(r)) for r in result[key])
              for key in ('train', 'val')}
    cohort_hash = fingerprint(cohort)
    snapshot = run/'motion_cohort.json'
    saved = json.loads(snapshot.read_text()) if snapshot.exists() else None
    if saved and saved['fingerprint'] != cohort_hash:
        if saved.get('label_policy') != LABEL_POLICY and not (run/'checkpoint.pt').exists():
            log.warning('Upgrade never-trained motion cohort after explicit label quarantine')
            saved = None
        else:
            raise ValueError('Frozen motion eligibility/matching changed; use a NEW run directory')
    if not saved:
        atomic_json(snapshot, dict(fingerprint=cohort_hash, cohorts=cohort, label_policy=LABEL_POLICY))
    report['cohort_hash'] = cohort_hash
    atomic_json(run/'motion_data_audit.json', report)
    return result


def estimate_flow(pixels, fps):
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3 or len(pixels) < 2:
        raise ValueError('Flow estimator needs at least two RGB uint8 frames')
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in pixels]
    fields = []
    for left, right in zip(gray, gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(left, right, None, **FLOW_PARAMS)
        flow -= np.median(flow.reshape(-1, 2), axis=0)
        # Fixed physical scale across videos, never divide by a per-clip maximum.
        flow *= fps / pixels.shape[2]
        if not np.isfinite(flow).all():
            raise RuntimeError('Nonfinite optical flow')
        fields.append(flow)
    return np.stack(fields).astype(np.float16)


def cache_flow(pixels, path, cache, fps):
    field = estimate_flow(pixels, fps)
    buffer = io.BytesIO()
    np.save(buffer, field, allow_pickle=False)
    cache.put(path, buffer.getvalue())
    return dict(mean_speed=float(np.linalg.norm(field.astype(np.float32), axis=-1).mean()),
                zero_neighbor_fraction=float(np.mean(np.all(pixels[1:] == pixels[:-1], axis=(1, 2, 3)))))


def tensor_from_flow(field, size, scale, seed=None, stress=False, full_frame=False):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Flow scale must be positive and finite')
    if field.ndim != 4 or field.shape[-1] != 2 or not np.isfinite(field).all():
        raise ValueError('Expected finite T,H,W,2 velocity field')
    rng = random.Random(seed)
    side = field.shape[1]
    training = seed is not None and not stress
    crop = rng.randint(size, side) if training and not full_frame else side
    left, top = (rng.randint(0, side-crop) for _ in range(2)) if training and not full_frame else (0, 0)
    flip = training and rng.random() < .5
    frames = []
    for raw in field:
        value = raw[top:top+crop, left:left+crop].astype(np.float32).copy()
        value = cv2.resize(value, (size, size), interpolation=cv2.INTER_LINEAR) * (side/crop)
        if flip:
            value = value[:, ::-1].copy()
            value[..., 0] *= -1
        if stress:
            value = cv2.GaussianBlur(value, (5, 5), 0)
            value = np.round(value*1024)/1024
        frames.append(value)
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).div(scale).clamp(-1, 1)


def flow_colors(x):
    fields = x.detach().cpu().permute(1, 2, 3, 0).numpy()
    output = []
    for field in fields:
        hsv = np.zeros((*field.shape[:2], 3), np.uint8)
        hsv[..., 0] = ((np.arctan2(field[..., 1], field[..., 0])+np.pi)*179/(2*np.pi)).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = (np.clip(np.linalg.norm(field, axis=-1), 0, 1)*255).astype(np.uint8)
        output.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    return np.stack(output)


def motion_variant(x, variant, seed=42):
    if variant == 'zero':
        return torch.zeros_like(x)
    if variant == 'mean_field':
        return x.mean(2, keepdim=True).expand_as(x)
    if variant == 'shuffle':
        order = torch.randperm(x.shape[2], generator=torch.Generator().manual_seed(seed)).to(x.device)
        return x[:, :, order]
    raise ValueError('Unknown motion control')
