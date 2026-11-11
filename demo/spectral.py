"""Optional offline inference using the frozen spectral-face-v3 experiment."""

import json
import math

import cv2

from training.spectral.faces import MODEL, NAME, SHA256
from training.spectral.features import FACE_RECIPE, extract
from training.spectral.model import scores_for
from training.spectral.run import FACE_RUN, versions
from .inference import ROOT, sha256


def assets(prefer_training=False):
    bundle = ROOT / 'demo_weights'
    frozen = bundle / 'spectral.json', bundle / NAME
    training = FACE_RUN / 'model.json', MODEL
    if prefer_training and all(path.is_file() for path in training):
        return training
    if all(path.is_file() for path in frozen):
        return frozen
    return training


class SpectralDetector:
    def __init__(self):
        model, self.faces = assets()
        self.saved = json.loads(model.read_text(encoding='utf-8'))
        config = self.saved['config']
        if (config['recipe'] != FACE_RECIPE or config['versions'] != versions() or
                config['yunet_sha256'] != SHA256 or sha256(self.faces) != SHA256):
            raise ValueError('Incompatible spectral weights, decoder or YuNet checksum')
        if 'full_face' not in self.saved['models'] or config['frames'] != 3:
            raise ValueError('Unsupported spectral recipe configuration')
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
