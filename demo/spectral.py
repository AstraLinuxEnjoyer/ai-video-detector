"""Optional offline inference using the frozen spectral-face-v3 experiment."""

import json
import math
import shutil
import subprocess

import cv2

from training.spectral.faces import MODEL, NAME, SHA256
from training.spectral.features import FACE_RECIPE, extract
from training.spectral.model import scores_for
from training.spectral.run import FACE_RUN, versions
from .inference import ROOT, sha256
from .runtime import configure_tools


def assets(prefer_training=False):
    bundle = ROOT / 'demo_weights'
    frozen = bundle / 'spectral.json', bundle / NAME
    training = FACE_RUN / 'model.json', MODEL
    if prefer_training and all(path.is_file() for path in training):
        return training
    if all(path.is_file() for path in frozen):
        return frozen
    return training


def spectral_status():
    """Check the same frozen contract used at inference, with actionable errors."""
    configure_tools()
    model, faces = assets()
    result = dict(status="unavailable", recipe=None, sha256=None, issues=[],
                  expected_versions={}, actual_versions={},
                  paths=dict(model=str(model), yunet=str(faces),
                             ffmpeg=shutil.which("ffmpeg"), ffprobe=shutil.which("ffprobe")))
    issues = result["issues"]
    for title, path in (("Spectral model", model), ("YuNet weights", faces)):
        if not path.is_file():
            issues.append(f"{title} missing: {path}. Restore the complete demo_weights bundle.")
    if issues:
        result["message"] = "; ".join(issues)
        return result
    try:
        saved = json.loads(model.read_text(encoding="utf-8"))
        config = saved["config"]
        if not isinstance(config, dict) or not isinstance(config.get("versions"), dict):
            raise ValueError("config.versions must be a mapping")
        result.update(recipe=config.get("recipe"), sha256=sha256(model),
                      expected_versions=config["versions"])
        if config.get("recipe") != FACE_RECIPE:
            issues.append(f"Spectral recipe: expected {FACE_RECIPE!r}; found {config.get('recipe')!r}.")
        try:
            actual = versions()
            result["actual_versions"] = actual
            for key in sorted(set(config["versions"]) | set(actual)):
                expected, found = config["versions"].get(key), actual.get(key)
                if expected != found:
                    title = {"ffmpeg":"FFmpeg", "numpy":"NumPy", "opencv":"OpenCV"}.get(key, key)
                    issues.append(f"{title}: expected {expected!r}; found {found!r}.")
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            issues.append(f"Cannot inspect FFmpeg/runtime: {exc}")
        if not result["paths"]["ffprobe"]:
            issues.append("FFprobe missing. Install the matching ffmpeg/ffprobe pair in tools/.")
        if config.get("yunet_sha256") != SHA256:
            issues.append(f"YuNet metadata checksum: expected {SHA256}; found {config.get('yunet_sha256')}.")
        face_sha = sha256(faces)
        if face_sha != SHA256:
            issues.append(f"YuNet file checksum: expected {SHA256}; found {face_sha}. Restore the original ONNX file.")
        if "full_face" not in saved.get("models", {}) or config.get("frames") != 3:
            issues.append("Unsupported spectral configuration: full_face and exactly 3 frames are required.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(f"Cannot validate spectral assets: {exc}")
    if issues:
        issues.append("Use the FFmpeg build recorded in spectral.json in the project's tools/ folder, and pinned demo-requirements.txt. Do not edit the model metadata to bypass validation.")
    result["status"] = "ready" if not issues else "unavailable"
    result["message"] = "; ".join(issues) if issues else "Spectral model and preprocessing are compatible."
    return result


class SpectralDetector:
    def __init__(self):
        state = spectral_status()
        if state['status'] != 'ready':
            raise ValueError('Spectral model unavailable: ' + state['message'])
        model, self.faces = assets()
        self.saved = json.loads(model.read_text(encoding='utf-8'))
        self.info = dict(name='Spectral / face v3', recipe=FACE_RECIPE, sha256=sha256(model))

    def analyze(self, source, job, directory):
        samples = []

        def preview(timestamp, gray, bands, box):
            job.check()
            h, w = gray.shape
            scale = min(1., 960 / max(h, w))
            image = cv2.resize(gray, (max(1, round(w*scale)), max(1, round(h*scale))),
                               interpolation=cv2.INTER_AREA) if scale < 1 else gray
            if box is not None:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                a, b, c, d = box
                cv2.rectangle(image, (round(a*scale), round(b*scale)),
                              (round(c*scale)-1, round(d*scale)-1), (127, 236, 193), 2)
            name = f'spectral_{len(samples):02d}.jpg'
            ok, jpg = cv2.imencode('.jpg', image)
            if not ok:
                raise RuntimeError('Cannot encode spectral preview')
            (directory / name).write_bytes(jpg.tobytes())
            samples.append(dict(time=timestamp, image=name, bands_db=(10*bands).tolist(), face=box is not None))

        job.check()
        row = extract(source, 3, recipe=FACE_RECIPE, face_weights=self.faces, preview=preview)
        models = self.saved['models']
        variants = {}
        for name, model in models.items():
            score = float(scores_for(models, name, [row])[0])
            variants[name] = dict(score=score if math.isfinite(score) else None, threshold=model['threshold'])
        if variants['full_face']['score'] is None:
            raise ValueError('Spectral model returned no usable score')
        return dict(status='complete', **variants['full_face'], samples=samples, variants=variants,
                    face_frames=row['face_counts']['clean'])
