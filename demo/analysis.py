import csv
import json
import logging
import math
import re
import subprocess
import time
from pathlib import Path

from training.spatial.common import atomic_json

LOG = logging.getLogger('trace')
MAX_SECONDS = 120
FRAME_STEP = 2.
THRESHOLD = .65


class Cancelled(Exception):
    pass


def segments(frames, clips, duration, threshold=THRESHOLD):
    spans = [(max(0., f['time']), min(duration, f['time']+FRAME_STEP)) for f in frames
             if f.get('spatial', 0) >= threshold and 0 <= f['time'] < duration]
    spans += [(max(0., c['start']), min(duration, c['end'])) for c in clips
              if c.get('temporal', 0) >= threshold and c['start'] < duration and c['end'] > 0]
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1]['end']+.05:
            merged[-1]['end'] = max(merged[-1]['end'], end)
        else:
            merged.append(dict(start=start, end=end))
    return merged


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(path)], capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError('File is not a readable video')
    payload = json.loads(result.stdout)
    stream = next((s for s in payload['streams'] if s.get('codec_type') == 'video'), None)
    if stream is None:
        raise ValueError('No video track in file')
    duration = float(stream.get('duration') or payload.get('format', {}).get('duration') or 0)
    numerator, denominator = map(float, stream.get('avg_frame_rate', '0/1').split('/'))
    fps = numerator/denominator if denominator else 0
    if not math.isfinite(duration) or not 0 < duration <= MAX_SECONDS:
        raise ValueError(f'Demo supports videos up to {MAX_SECONDS} seconds; trim a copy of the original')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Cannot determine native frame rate')
    return dict(duration=duration, fps=fps, width=stream['width'], height=stream['height'],
                codec=stream.get('codec_name'), transfer=stream.get('color_transfer'))


def filters(meta):
    if meta.get('transfer') in {'smpte2084', 'arib-std-b67'}:
        return ['zscale=t=linear:npl=100', 'format=gbrpf32le', 'zscale=p=bt709',
                'tonemap=tonemap=hable:desat=0', 'zscale=t=bt709:m=bt709:r=full']
    return []


def ffmpeg(command, job, directory, timeout=180):
    # A log file avoids a full stderr pipe deadlocking video conversion.
    with (directory/'ffmpeg.log').open('ab') as stream:
        process = subprocess.Popen(['ffmpeg', '-hide_banner', '-nostdin', '-y', '-threads', '2']+command,
                                   stdout=subprocess.DEVNULL, stderr=stream)
        started = time.monotonic()
        try:
            while process.poll() is None:
                job.check()
                if time.monotonic()-started > timeout:
                    raise TimeoutError('Video decoding timed out')
                time.sleep(.15)
            if process.returncode:
                raise RuntimeError('FFmpeg could not decode video; see ffmpeg.log in the local job folder')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def analyze(job, detector):
    directory, source = job.directory, job.directory/'input.video'
    started = time.monotonic()
    job.update(phase='decode', progress=5, message='Читаем видео и извлекаем кадры')
    meta = probe(source)
    job.update(metadata=meta)
    image_filters = filters(meta)+[f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{FRAME_STEP})'",
                                    'scale=960:960:force_original_aspect_ratio=decrease', 'showinfo']
    ffmpeg(['-i', str(source), '-an', '-sn', '-vf', ','.join(image_filters), '-filter_threads', '1',
            '-fps_mode', 'vfr', '-q:v', '2', str(directory/'frame_%04d.jpg')], job, directory)
    times = [float(t) for t in re.findall(r'\bpts_time:([\d.eE+\-]+)',
                                        (directory/'ffmpeg.log').read_text(errors='replace'))]
    paths = sorted(directory.glob('frame_*.jpg'))
    if not paths or len(paths) != len(times):
        raise RuntimeError('Frame timestamps could not be verified')
    frames = [dict(index=i, time=t, image=f'frame_{i+1:04d}.jpg') for i, t in enumerate(times)]
    job.update(frames=frames, progress=15, message=f'Извлечено кадров: {len(frames)}')
    # Only this playback copy is transcoded. Models read the original video/decoded originals.
    ffmpeg(['-i', str(source), '-an', '-sn', '-vf', ','.join(filters(meta)+[
            'scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2', 'setsar=1']),
            '-filter_threads', '1', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(directory/'preview.mp4')], job, directory)
    job.update(video='preview.mp4', progress=20)
    job.update(phase='spatial', message='Первая модель: общий кадр и локальные участки')
    for i, frame in enumerate(frames):
        job.check()
        frame.update(detector.spatial_score(directory/frame['image']))
        job.update(frames=frames, progress=20+35*(i+1)/len(frames), done=i+1, total=len(frames))
        LOG.info('%s frame=%d t=%.3f spatial=%.5f global=%.5f local=%.5f',
                 job.id, i, frame['time'], frame['spatial'], frame['global_score'], frame['local_score'])
    clips, warnings = [], []
    args = detector.args
    seconds = args.clip_length/args.fps
    if meta['duration'] < seconds or meta['fps'] < args.fps:
        warnings.append('Модель движения не применена: слишком короткое видео или исходный FPS ниже 4. Кадры оценены первой моделью.')
    else:
        count = max(1, math.ceil((meta['duration']-seconds)/1.5)+1)
        positions = [min(i*1.5, meta['duration']-seconds) for i in range(count)]
        job.update(phase='temporal', progress=55, message='Вторая модель: последовательности и оптический поток')
        for i, start in enumerate(positions):
            job.check()
            flow = f'flow_{i:04d}.gif'
            try:
                score = detector.motion_score(source, start, meta, directory/flow)
                clips.append(dict(index=i, start=start, end=start+seconds, temporal=score, flow=flow))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                warnings.append(f'Клип {start:.2f}–{start+seconds:.2f} с не прочитан; оценка отсутствует.')
                LOG.exception('%s failed clip %s: %s', job.id, start, exc)
            job.update(clips=clips, warnings=warnings, progress=55+30*(i+1)/len(positions), done=i+1, total=len(positions))
            if clips:
                LOG.info('%s motion t=%.3f score=%.5f', job.id, clips[-1]['start'], clips[-1]['temporal'])
    job.update(phase='explain', progress=85, message='Строим карты влияния и итоговую дорожку')
    selected = sorted(frames, key=lambda f: f['spatial'], reverse=True)[:4]
    for i, frame in enumerate(selected):
        job.check()
        heat = f'heat_{frame["index"]:04d}.jpg'
        valid = detector.heatmap(directory/frame['image'], directory/heat)
        frame.update(heatmap=heat, heat_valid=valid)
        job.update(frames=frames, progress=85+12*(i+1)/len(selected))
    spatial = sum(f['spatial'] for f in frames)/len(frames)
    temporal = sum(c['temporal'] for c in clips)/len(clips) if clips else None
    fused = .7*spatial+.3*temporal if temporal is not None else spatial
    regions = segments(frames, clips, meta['duration'])
    summary = dict(spatial=spatial, temporal=temporal, combined=fused, threshold=THRESHOLD,
                   spatial_samples=len(frames), temporal_samples=len(clips), segments=regions,
                   suspicious_seconds=sum(s['end']-s['start'] for s in regions),
                   disagreement=temporal is not None and abs(spatial-temporal) >= .35,
                   verdict='suspected' if regions else 'uncertain' if fused >= .35 else 'no_signals',
                   elapsed_seconds=time.monotonic()-started,
                   caveat='Оценки не являются вероятностями. Подсветка — диагностический сигнал, не доказательство. '
                          'Шаг кадров 2 с, окно движения 2.25 с; короткие вставки и локальные правки могут быть пропущены. '
                          'Grad-CAM полного кадра показывает влияние на fake-логит, а не маску генерации. '
                          'Смешивание 70/30 — демонстрационная эвристика, не обученный ансамбль.')
    with (directory/'frames.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['time', 'spatial', 'global_score', 'local_score'])
        writer.writeheader()
        writer.writerows({k: frame[k] for k in writer.fieldnames} for frame in frames)
    final = dict(job.snapshot(), status='complete', phase='complete', progress=100, summary=summary,
                 warnings=warnings, models=detector.describe(), message='Анализ завершен')
    atomic_json(directory/'report.json', final)
    job.update(**final)
    LOG.info('%s COMPLETE summary=%s', job.id, json.dumps(summary, ensure_ascii=False))
