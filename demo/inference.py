"""Inference contracts shared with the trained hybrid and motion experiments."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from torchvision.models import efficientnet_v2_s

from training.artifact_spatial.patches import aggregate_views, crop_boxes, load_image, view_tensor
from training.temporal_model import build_temporal_model
from training.temporal_v2.flow import estimate_flow, flow_colors, tensor_from_flow
from training.temporal_v2.media import decode_clip

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_sources(prefer_training=False):
    runs = {'spatial': ROOT/'cache/artifact_spatial/runs/hybrid_v2',
            'temporal': ROOT/'cache/temporal_v2/runs/motion_flow_v1'}
    final_motion = runs['temporal'].with_name('final_motion_v1')
    if (final_motion/'final_report.json').is_file():
        runs['temporal'] = final_motion
    bundle = ROOT/'demo_weights'
    manifest = bundle/'manifest.json'
    if manifest.is_file() and not (prefer_training and all((run/'best.pt').is_file() for run in runs.values())):
        info = json.loads(manifest.read_text(encoding='utf-8'))
        for kind in ('spatial', 'temporal'):
            entry = info[kind]
            path = (bundle/entry['file']).resolve()
            if path.parent != bundle.resolve() or not path.is_file():
                raise ValueError(f'Missing bundled {kind} weights')
            entry['path'] = path
            if kind == 'temporal':
                entry['run'] = (final_motion if entry['config']['recipe'] == 'temporal-final-motion-v1' else
                                final_motion.with_name('motion_flow_v1'))
            else:
                entry['run'] = runs[kind]
        return info
    info = {}
    for kind, run in runs.items():
        path = run/'best.pt'
        if not path.is_file():
            raise FileNotFoundError('Demo weights missing. Run export-demo.ps1 on the training computer.')
        config = json.loads((run/'config.json').read_text(encoding='utf-8'))
        best = json.loads((run/'best.json').read_text(encoding='utf-8'))
        info[kind] = dict(path=path, config=config, epoch=best['epoch'], sha256=sha256(path), run=run)
    return info


class Detector:
    def __init__(self, info=None):
        self.info = model_sources() if info is None else info
        self.device = torch.device(os.environ.get('TRACE_DEVICE', 'cpu'))
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('TRACE_DEVICE must be cpu or cuda')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use TRACE_DEVICE=cpu')
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        config = self.info['spatial']['config']
        if config.get('recipe') not in {'hybrid-v2', 'final-hybrid-v1'}:
            raise ValueError('Spatial detector requires hybrid weights and preprocessing')
        self.spatial_size = config['image_size']
        self.spatial = efficientnet_v2_s(weights=None)
        self.spatial.classifier[1] = torch.nn.Linear(self.spatial.classifier[1].in_features, 1)
        self._load(self.spatial, self.info['spatial'])
        config = self.info['temporal']['config']
        if config.get('input_mode') != 'flow' or config.get('recipe') not in {
                'temporal-motion-v1', 'temporal-final-motion-v1', 'temporal-motion-coverage-v1'}:
            raise ValueError('Temporal demo requires motion-flow weights, not RGB weights')
        self.args = SimpleNamespace(**config)
        if config.get('opencv_version') != cv2.__version__:
            raise ValueError('OpenCV version does not match the frozen flow recipe. Run setup-demo.cmd.')
        self.temporal = build_temporal_model(False, 'flow')
        self._load(self.temporal, self.info['temporal'])

    def _load(self, model, entry):
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('Weights checksum mismatch')
        model.load_state_dict(torch.load(entry['path'], map_location='cpu', weights_only=True), strict=True)
        model.to(self.device).eval().requires_grad_(False)

    def describe(self):
        return dict(device=str(self.device), torch_version=str(torch.__version__), opencv_version=cv2.__version__,
                    spatial=dict(name='EfficientNet V2-S / hybrid',
                    recipe=self.info['spatial']['config']['recipe'],
                    epoch=self.info['spatial']['epoch'], sha256=self.info['spatial']['sha256']),
                    temporal=dict(name='R(2+1)D-18 / motion flow',
                    recipe=self.info['temporal']['config']['recipe'],
                    epoch=self.info['temporal']['epoch'], sha256=self.info['temporal']['sha256']))

    def spatial_score(self, path):
        image = load_image(path)
        x = view_tensor(image, self.spatial_size, hybrid=True).to(self.device)
        with torch.inference_mode():
            scores = self.spatial(x).float().sigmoid().flatten().cpu()
        if not torch.isfinite(scores).all():
            raise RuntimeError('Nonfinite spatial output')
        return dict(spatial=float(aggregate_views(scores, True)), global_score=float(scores[0]),
                    local_score=float(scores[1:].mean()), patch_scores=scores[1:].tolist())

    def patch_preview(self, path, output, scores):
        image = load_image(path)
        boxes = crop_boxes(image, self.spatial_size)
        best = max(range(len(scores)), key=scores.__getitem__)
        x, y, side = boxes[best]
        draw = ImageDraw.Draw(image)
        draw.rectangle((x, y, x+side-1, y+side-1), outline='#ffb28c',
                       width=max(2, round(max(image.size)/190)))
        image.save(output, quality=91)
        return best

    def motion_score(self, path, start, meta, output=None):
        pixels = decode_clip(path, start, meta, self.args)
        field = estimate_flow(pixels, self.args.fps)
        x = tensor_from_flow(field, self.args.image_size, self.args.flow_scale,
                             full_frame=getattr(self.args, 'frame_view', 'crop') == 'full')
        with torch.inference_mode():
            score = self.temporal(x[None].to(self.device)).float().sigmoid().item()
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite temporal output')
        if output is not None:
            frames = [Image.fromarray(p).resize((224, 224)) for p in flow_colors(x)]
            frames[0].save(output, save_all=True, append_images=frames[1:], duration=250, loop=0)
        return float(score)

    def motion_influence(self, path, start, meta, output):
        pixels = decode_clip(path, start, meta, self.args)
        full = getattr(self.args, 'frame_view', 'crop') == 'full'
        field = estimate_flow(pixels, self.args.fps)
        x = tensor_from_flow(field, self.args.image_size, self.args.flow_scale, full_frame=full)
        edges = np.linspace(0, x.shape[-1], 4, dtype=int).tolist()
        boxes = [(a, b, c, d) for b, d in zip(edges, edges[1:]) for a, c in zip(edges, edges[1:])]
        batch = x[None].repeat(len(boxes)+1, 1, 1, 1, 1)
        for i, (a, b, c, d) in enumerate(boxes, 1):
            batch[i, :, :, b:d, a:c] = 0
        with torch.inference_mode():
            scores = torch.cat([self.temporal(rows.to(self.device)).flatten().sigmoid().cpu() for rows in batch.split(3)])
        if not torch.isfinite(scores).all():
            raise RuntimeError('Nonfinite motion influence response')
        drops = (scores[0]-scores[1:]).tolist()
        size = x.shape[-1]
        frame = Image.fromarray(pixels[len(pixels)//2])
        frame = frame.resize((336, 336), Image.Resampling.BILINEAR)
        overlay = Image.new('RGBA', frame.size)
        draw = ImageDraw.Draw(overlay)
        peak = max(drops)
        for i, ((a, b, c, d), drop) in enumerate(zip(boxes, drops)):
            if drop > 0:
                rect = tuple(round(k*336/size) for k in (a, b, c-1, d-1))
                draw.rectangle(rect, fill=(255, 126, 70, round(125*drop/peak)),
                               outline='#ffb28c' if i == drops.index(peak) else None, width=3)
        Image.alpha_composite(frame.convert('RGBA'), overlay).convert('RGB').save(output, quality=91)
        return dict(peak_drop=max(0., peak), drops=drops)

    def heatmap(self, path, output):
        # Attribute the fake logit of the global view, not the combined local score.
        image = load_image(path)
        x = view_tensor(image, self.spatial_size, hybrid=True)[:1].to(self.device).requires_grad_(True)
        captured = []
        hook = self.spatial.features[-1].register_forward_hook(lambda _m, _i, value: captured.append(value))
        try:
            with torch.enable_grad():
                logit = self.spatial(x).flatten()[0]
                gradient, = torch.autograd.grad(logit, captured[0])
                heat = (captured[0]*gradient.mean((2, 3), keepdim=True)).sum(1).relu()[0]
                peak = float(heat.max().detach())
                heat = heat.detach().cpu().numpy()/max(peak, 1e-12)
            heat = np.asarray(Image.fromarray(heat).resize(image.size, Image.Resampling.BILINEAR))
            rgb = np.asarray(image).astype(np.float32)
            color = np.zeros_like(rgb)
            color[..., 0], color[..., 1], color[..., 2] = 255, 75, 65
            alpha = heat[..., None]*.65
            Image.fromarray(np.uint8(rgb*(1-alpha)+color*alpha)).save(output, quality=90)
            return peak > 1e-12
        finally:
            hook.remove()
