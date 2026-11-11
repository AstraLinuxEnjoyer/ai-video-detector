"""Native-grid technical features; no RGB tensors or video metadata in the model."""

import base64
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from training.spatial.media import SourceChangedError, probe
from .faces import locate

PATCH_RECIPE = "spectral-residual-v1"
RECIPE = "spectral-full-frame-v2"
FACE_RECIPE = "spectral-face-v3"
PATCH = 128
BANDS = np.array([0, .05, .10, .15, .20, .25, .30, .35, .40, .50, .71])
SPECTRAL_DIM = 2 * (len(BANDS) - 1)
NOISE_DIM = 10
DIM = SPECTRAL_DIM + NOISE_DIM
WINDOW = np.outer(np.hanning(PATCH), np.hanning(PATCH))
RADIUS = np.hypot(np.fft.fftfreq(PATCH)[:, None], np.fft.fftfreq(PATCH)[None, :])
RING = np.digitize(RADIUS.ravel(), BANDS) - 1
VIEWS = ((.5, .5), (.25, .25), (.25, .75), (.75, .25), (.75, .75))
MAX_PIXELS = 4096 * 2304


def verify_record(record):
    path = Path(record["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime"]):
        raise SourceChangedError(f"Source changed since snapshot: {path}")


def residual_stats(x):
    residual = x - cv2.GaussianBlur(x, (5, 5), 1.2)
    magnitude = np.abs(residual)
    variance = float(np.mean(residual**2))
    stats = (magnitude.mean(), np.sqrt(variance), np.quantile(magnitude, .90),
             np.mean(residual[:, :-1] * residual[:, 1:]) / max(variance, 1e-10),
             np.mean(residual[:-1] * residual[1:]) / max(variance, 1e-10))
    return np.asarray(stats)


def log_spectrum(energy):
    return np.log10(np.maximum(energy / max(float(energy.sum()), 1e-12), 1e-8))


def patch_features(gray):
    if gray.shape != (PATCH, PATCH):
        raise ValueError(f"Expected {PATCH}x{PATCH} luma patch")
    x = gray.astype(np.float64) / 255
    power = np.abs(np.fft.fft2((x - x.mean()) * WINDOW)) ** 2
    power[0, 0] = 0
    energy = np.bincount(RING, weights=power.ravel(), minlength=len(BANDS)-1)
    return log_spectrum(energy), residual_stats(x)


@lru_cache(maxsize=4)
def frequency_rings(height, width):
    radius = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :])
    return np.digitize(radius.ravel(), BANDS) - 1


def full_frame_features(gray):
    height, width = gray.shape
    if min(height, width) < 16 or height*width > MAX_PIXELS:
        raise ValueError(f"Unsupported native dimensions: {width}x{height}")
    x = gray.astype(np.float32) / 255
    windowed = (x - x.mean()) * np.hanning(height).astype(np.float32)[:, None]
    windowed *= np.hanning(width).astype(np.float32)[None, :]
    power = np.abs(np.fft.rfft2(windowed)) ** 2
    power[0, 0] = 0
    # Restore the energy of omitted conjugate columns in the real FFT.
    power[:, 1:None if width % 2 else -1] *= 2
    energy = np.bincount(frequency_rings(height, width), weights=power.ravel(), minlength=len(BANDS)-1)
    return log_spectrum(energy), residual_stats(x)


def frame_patches(gray):
    height, width = gray.shape
    if min(height, width) < PATCH:
        raise ValueError(f"Native frame is smaller than {PATCH}px; not upsampling")
    positions = dict.fromkeys((round((height-PATCH)*y), round((width-PATCH)*x)) for y, x in VIEWS)
    return [patch_features(gray[y:y+PATCH, x:x+PATCH]) for y, x in positions]


def pool_features(patches):
    spectrum, noise = map(np.stack, zip(*patches))
    vector = np.concatenate([values for block in (spectrum, noise)
                             for values in (block.mean(axis=0), np.quantile(block, .9, axis=0))])
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise ValueError("Invalid spectral feature vector")
    return vector.tolist()


def decode_luma(path, timestamp, meta, recipe=RECIPE):
    width, height = meta["width"], meta["height"]
    minimum = PATCH if recipe == PATCH_RECIPE else 16
    if min(width, height) < minimum or width * height > MAX_PIXELS:
        raise ValueError(f"Unsupported native dimensions: {width}x{height}")
    if meta["transfer"] in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR outside spectral recipe; no tone mapping in spectral extraction")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
               "-noautorotate", "-ss", f"{timestamp:.6f}", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-filter_threads", "1",
               "-frames:v", "1", "-pix_fmt", "gray", "-c:v", "rawvideo",
               "-threads", "1", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(command, capture_output=True, timeout=45)
    if result.returncode or len(result.stdout) != width * height:
        raise ValueError(result.stderr.decode(errors="replace")[-800:] or "Empty or unexpected luma frame")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width)


def jpeg_stress(gray):
    ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise ValueError("JPEG stress encode failed")
    result = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if result is None or result.shape != gray.shape:
        raise ValueError("JPEG stress decode failed")
    return result


def frame_views(gray, recipe):
    if recipe in (RECIPE, FACE_RECIPE):
        return [full_frame_features(gray)]
    if recipe == PATCH_RECIPE:
        return frame_patches(gray)
    raise ValueError(f"Unknown spectral recipe: {recipe}")


def face_features(gray, weights):
    box = locate(gray, weights)
    if box is None:
        return None, None
    x0, y0, x1, y1 = box
    return full_frame_features(gray[y0:y1, x0:x1]), list(box)


def extract(path, frames, stress=False, recipe=RECIPE, face_weights=None, preview=None):
    meta = probe(path)
    end = max(0, meta["duration"] - max(.1, 2/max(meta["fps"], 1)))
    positions = [end * (i+.5)/frames for i in range(frames)]
    clean_patches, stress_patches, faces, stress_faces, boxes = [], [], [], [], []
    if recipe == FACE_RECIPE and face_weights is None:
        raise ValueError("The face recipe requires verified YuNet weights")
    for timestamp in positions:
        gray = decode_luma(path, timestamp, meta, recipe)
        views = frame_views(gray, recipe)
        clean_patches.extend(views)
        box = None
        if face_weights is not None:
            feature, box = face_features(gray, face_weights)
            boxes.append(box)
            if feature is not None:
                faces.append(feature)
        if preview is not None:
            preview(timestamp, gray, views[0][0], box)
        if stress:
            changed = jpeg_stress(gray)
            stress_patches.extend(frame_views(changed, recipe))
            if face_weights is not None:
                feature, _ = face_features(changed, face_weights)
                if feature is not None:
                    stress_faces.append(feature)
    vectors = {"clean": pool_features(clean_patches)}
    if stress:
        vectors["jpeg70"] = pool_features(stress_patches)
    result = dict(vectors=vectors, timestamps=positions, patches=len(clean_patches), preview_meta=meta)
    if face_weights is not None:
        result.update(face_boxes=boxes, face_counts=dict(clean=len(faces), jpeg70=len(stress_faces)),
                      face_vectors={t: pool_features(items) for t, items in
                                    (("clean", faces), ("jpeg70", stress_faces)) if items})
    return result


def preview_frame(record, recipe):
    path = Path(record["path"])
    timestamp = record["timestamps"][len(record["timestamps"])//2]
    meta = record.get("preview_meta") or probe(path)
    gray = decode_luma(path, timestamp, meta, recipe)
    spectrum = np.stack([item[0] for item in frame_views(gray, recipe)]).mean(axis=0).tolist()
    index = len(record["timestamps"])//2
    box = record["face_boxes"][index] if record.get("face_boxes") else None
    if max(gray.shape) > 640:
        scale = 640/max(gray.shape)
        gray = cv2.resize(gray, (max(1, round(gray.shape[1]*scale)), max(1, round(gray.shape[0]*scale))),
                          interpolation=cv2.INTER_AREA)
    if box is not None:
        x0, y0, x1, y1 = box
        sx, sy = gray.shape[1]/meta["width"], gray.shape[0]/meta["height"]
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(gray, (round(x0*sx), round(y0*sy)), (round(x1*sx)-1, round(y1*sy)-1), (93, 218, 87), 2)
    ok, jpeg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Cannot encode dashboard preview")
    return dict(name=path.name, source=record["source"], label="FAKE" if record["label"] else "REAL",
                timestamp=timestamp, width=meta["width"], height=meta["height"], spectrum=spectrum,
                image=base64.b64encode(jpeg.tobytes()).decode("ascii"), face_box=box,
                face_frames=record.get("face_counts", {}).get("clean"),
                frames=len(record["timestamps"]))
