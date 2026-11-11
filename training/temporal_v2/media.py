"""Decode complete clips before GPU work; never substitute another video."""
import io
import json
import random
import subprocess
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from training.spatial.common import atomic_json, event, fingerprint
from training.spatial.media import FrameCache, SourceChangedError, decoding_pool, probe
from training.spatial.sequences import sequence_unchanged
from .flow import (FLOW_PARAMS, FLOW_VERSION, MotionInputError, cache_flow, check_metadata,
                   flow_colors, is_motion, tensor_from_flow)

MEAN = (.43216, .394666, .37645)
STD = (.22803, .22145, .216989)


class ClipTooShortError(ValueError):
    """Valid media outside the fixed temporal input contract."""

    def __init__(self, duration, required):
        self.duration, self.required = duration, required
        super().__init__(f"Clip too short: {duration:.3f}s, need {required:.3f}s; no frame repetition")


def starts(duration, length, fps, count, seed, training):
    if duration < length / fps:
        raise ClipTooShortError(duration, length / fps)
    end = max(0., duration - length / fps)
    rng = random.Random(seed)
    return [end * (i + rng.random()) / count if training else end * (i+.5)/count for i in range(count)]


def decode_clip(path, start, meta, args):
    filters = []
    if meta.get('transfer') in {'smpte2084', 'arib-std-b67'}:
        filters += ['zscale=t=linear:npl=100', 'format=gbrpf32le', 'zscale=p=bt709',
                    'tonemap=tonemap=hable:desat=0', 'zscale=t=bt709:m=bt709:r=full']
    # One sequential decode, one uniform time grid, same geometry for both classes.
    filters += [f'fps={args.fps}']
    if getattr(args, 'frame_view', 'crop') == 'full':
        filters += [f'scale={args.decode_size}:{args.decode_size}']
    else:
        filters += [f'scale={args.decode_size}:{args.decode_size}:force_original_aspect_ratio=increase',
                    f'crop={args.decode_size}:{args.decode_size}']
    filters += ['setsar=1', 'format=rgb24']
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1',
               '-ss', f'{start:.6f}', '-i', str(path), '-an', '-sn', '-dn',
               '-vf', ','.join(filters), '-filter_threads', '1', '-frames:v', str(args.clip_length),
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1']
    result = subprocess.run(command, capture_output=True, timeout=90)
    expected = args.clip_length * args.decode_size**2 * 3
    if result.returncode or len(result.stdout) != expected:
        raise RuntimeError(f"Incomplete clip {len(result.stdout)}/{expected} bytes: "
                           + result.stderr.decode(errors='replace')[-1000:])
    return np.frombuffer(result.stdout, np.uint8).reshape(args.clip_length, args.decode_size, args.decode_size, 3).copy()


def prepare(records, cache, tag, count, args, log, run, publish=None, *, allow_short=False):
    if allow_short and tag not in {'test', 'ood', 'external'}:
        raise ValueError('Short-video abstentions are allowed only in held-out evaluation')
    training = tag.startswith('train_')

    def one(record):
        path = Path(record['path'])
        sequence = record.get('kind') == 'image_sequence'
        unchanged = sequence_unchanged(record) if sequence else (
            (path.stat().st_size, path.stat().st_mtime_ns) == (record['size'], record['mtime']))
        if not unchanged:
            raise SourceChangedError(f"Frozen source changed: {path}")
        directory = cache.root / tag / record['id']
        manifest = directory / 'clips.json'
        signature = fingerprint([record['id'], record['mtime'], tag, count, args.seed,
                                 args.fps, args.clip_length, args.decode_size, args.sequence_fps]
                                + (['full-frame'] if getattr(args, 'frame_view', 'crop') == 'full' else [])
                                + ([FLOW_VERSION, FLOW_PARAMS, __import__('cv2').__version__] if is_motion(args) else []))
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['signature'] == signature and all(Path(p).is_file() for p in saved['clips'] + saved.get('flow_clips', [])):
                return dict(record, **saved)
        if sequence:
            if args.sequence_fps <= 0:
                raise ValueError('Image sequences require an explicitly supplied sequence FPS')
            meta = dict(duration=len(record['frame_files'])/args.sequence_fps, fps=args.sequence_fps,
                        width=record['sequence_width'], height=record['sequence_height'], codec='image_sequence')
        else:
            meta = probe(path)
        if is_motion(args):
            check_metadata(meta, args)
        positions = starts(meta['duration'], args.clip_length, args.fps, count,
                           f"{args.seed}:{tag}:{record['id']}", training)
        clips, repeated, recoveries, flow_clips, flow_stats = [], [], [], [], []
        for index, start in enumerate(positions):
            if sequence:
                indices = [int((start + i/args.fps)*args.sequence_fps) for i in range(args.clip_length)]
                if len(set(indices)) != len(indices):
                    raise ValueError('Sequence FPS would require repeated frames')
                frames = []
                for frame_index in indices:
                    with Image.open(path / record['frame_files'][frame_index]) as image:
                        frames.append(np.array(ImageOps.fit(image.convert('RGB'), (args.decode_size,)*2)))
                pixels = np.stack(frames)
            else:
                try:
                    pixels = decode_clip(path, start, meta, args)
                except RuntimeError as original:
                    # Some containers overstate their playable tail. Retry the SAME
                    # video slightly earlier, with recorded actual timestamps.
                    recovered = False
                    for shift in (.25, .5):
                        retry_start = max(0., start-shift)
                        if retry_start == start:
                            continue
                        try:
                            pixels = decode_clip(path, retry_start, meta, args)
                            recoveries.append(dict(clip=index, requested=start, actual=retry_start, reason=str(original)))
                            positions[index] = retry_start
                            recovered = True
                            break
                        except RuntimeError:
                            pass
                    if not recovered:
                        raise original
            repeated.append(float(np.mean(np.all(pixels[1:] == pixels[:-1], axis=(1, 2, 3)))))
            buffer = io.BytesIO()
            np.save(buffer, pixels, allow_pickle=False)
            output = directory / f'{index:02}.npy'
            cache.put(output, buffer.getvalue())
            clips.append(str(output))
            if is_motion(args):
                flow_path = directory / f'{index:02}_flow.npy'
                flow_stats.append(cache_flow(pixels, flow_path, cache, args.fps))
                flow_clips.append(str(flow_path))
        saved = dict(signature=signature, clips=clips, starts=positions, metadata=meta,
                     fps=args.fps, seconds=args.clip_length/args.fps,
                     repeated_adjacent_fraction=repeated, seek_recoveries=recoveries)
        if is_motion(args):
            saved.update(flow_clips=flow_clips, flow_stats=flow_stats, input_mode='flow')
        cache.put(manifest, json.dumps(saved).encode())
        return dict(record, **saved)

    results, failures, unsupported = [], [], []
    with decoding_pool(args.decode_workers) as executor:
        futures = {executor.submit(one, row): row for row in records}
        for index, future in enumerate(tqdm(as_completed(futures), total=len(records), desc=f'DECODE {tag}'), 1):
            record = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                detail = dict(id=record['id'], source=record['source'], label=record['label'],
                              path=record['path'], phase=tag, error=str(exc))
                if isinstance(exc, ClipTooShortError):
                    detail.update(reason='clip_too_short', duration=exc.duration, required_seconds=exc.required)
                if isinstance(exc, MotionInputError):
                    detail.update(reason='native_fps_below_target')
                failures.append(detail)
                event(run/'data_errors.jsonl', detail)
                if allow_short and isinstance(exc, (ClipTooShortError, MotionInputError)):
                    unsupported.append(detail)
                    log.warning('DECODE UNSUPPORTED %s; no prediction, no replacement', detail)
                else:
                    log.error('DECODE %s', detail)
                if isinstance(exc, (SourceChangedError, OSError)):
                    raise
            if publish:
                publish(phase=f'decode/{tag}', done=index, total=len(records), decode_errors=len(failures),
                        cache_gb=cache.used/1024**3)
    atomic_json(run/f'{tag}_decode.json', dict(requested=len(records), usable=len(results), errors=failures,
                unsupported=unsupported,
                native_fps_below_target=sum(r['metadata'].get('fps', 0) < args.fps for r in results),
                clips_with_identical_neighbors=sum(v > 0 for r in results for v in r.get('repeated_adjacent_fraction', [])),
                recovered_seeks=[dict(id=r['id'], **recovery) for r in results for recovery in r.get('seek_recoveries', [])],
                caveat='Uniform output FPS can duplicate low-FPS frames. Static scenes can also have equal frames; '
                       'these counts are diagnostics, not fake labels.'))
    fatal_count = len(failures)-len(unsupported)
    if fatal_count:
        raise RuntimeError(f'{fatal_count} videos failed in {tag}; no replacement or silent skipping. See data_errors.jsonl')
    recovered = sum(len(r.get('seek_recoveries', [])) for r in results)
    log.info('DECODE %s requested=%d usable=%d unsupported=%d earlier_seek_recoveries=%d cache=%.2f GiB',
             tag, len(records), len(results), len(unsupported), recovered, cache.used/1024**3)
    return sorted(results, key=lambda r: r['id'])


def tensor_from_pixels(pixels, size, seed=None, stress=False):
    rng = random.Random(seed)
    side = pixels.shape[1]
    training = seed is not None and not stress
    crop = rng.randint(size, side) if training else side
    left = rng.randint(0, side-crop) if training else 0
    top = rng.randint(0, side-crop) if training else 0
    flip = training and rng.random() < .5
    brightness = rng.uniform(.9, 1.1) if training else 1.
    degrade = stress or (training and rng.random() < .4)
    quality = 65 if stress else rng.randint(55, 95)
    shrink = .65 if stress else rng.uniform(.65, 1.)
    # Simulate frame holding in BOTH classes. Stress equalizes the output pattern
    # to four distinct time positions per second, not interpolated motion.
    hold = 4 if stress else rng.choice((1, 1, 2, 4, 4, 4)) if training else 1
    pixels = pixels[(np.arange(len(pixels))//hold)*hold]
    frames = []
    for raw in pixels:
        image = Image.fromarray(raw).crop((left, top, left+crop, top+crop)).resize((size, size), Image.Resampling.BILINEAR)
        if flip:
            image = ImageOps.mirror(image)
        if training:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if degrade:
            image = image.resize((round(size*shrink),)*2).resize((size,)*2)
            buffer = io.BytesIO()
            image.save(buffer, 'JPEG', quality=quality)
            with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
                image = decoded.convert('RGB')
        frames.append(np.array(image))
    x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float()/255.
    return (x-torch.tensor(MEAN)[:, None, None, None])/torch.tensor(STD)[:, None, None, None]


def pixels_from_tensor(x):
    if x.shape[0] == 2:
        return flow_colors(x)
    x = x.detach().cpu()*torch.tensor(STD)[:, None, None, None]+torch.tensor(MEAN)[:, None, None, None]
    return (x.clamp(0, 1).permute(1, 2, 3, 0)*255).byte().numpy()


class Clips(Dataset):
    def __init__(self, records, size, seed=None, stress=False, input_mode='rgb', flow_scale=.5, full_frame=False):
        self.items = [(r, i) for r in records for i in range(len(r['clips']))]
        self.size, self.seed, self.stress = size, seed, stress
        self.input_mode, self.flow_scale = input_mode, flow_scale
        self.full_frame = full_frame
        if seed is not None:
            random.Random(seed).shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record, clip = self.items[index]
        paths = record['flow_clips'] if self.input_mode == 'flow' else record['clips']
        pixels = np.load(paths[clip], allow_pickle=False)
        seed = None if self.seed is None else f"{self.seed}:{record['id']}:{clip}"
        x = (tensor_from_flow(pixels, self.size, self.flow_scale, seed, self.stress, self.full_frame)
             if self.input_mode == 'flow' else tensor_from_pixels(pixels, self.size, seed, self.stress))
        return x, float(record['label']), index


def clips_for(records, args, seed=None, stress=False):
    return Clips(records, args.image_size, seed, stress, getattr(args, 'input_mode', 'rgb'),
                 getattr(args, 'flow_scale', .5), getattr(args, 'frame_view', 'crop') == 'full')
