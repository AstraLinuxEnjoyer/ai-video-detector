"""Diagnostic attribution, not a causal explanation or training objective."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from training.spatial.common import atomic_json, replace_file
from .media import pixels_from_tensor
from .flow import is_motion, motion_variant


def save_gif(path, frames, fps):
    images = [Image.fromarray(frame).resize((224, 224)) for frame in frames]
    tmp = path.with_suffix('.tmp')
    images[0].save(tmp, format='GIF', save_all=True, append_images=images[1:], duration=round(1000/fps), loop=0)
    replace_file(tmp, path)


def save_strip(path, frames):
    sheet = Image.new('RGB', (8*112, ((len(frames)+7)//8)*136), '#111111')
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        x, y = (i%8)*112, (i//8)*136
        sheet.paste(Image.fromarray(frame).resize((112, 112)), (x, y))
        draw.text((x+4, y+115), str(i+1), fill='white')
    tmp = path.with_suffix('.tmp')
    sheet.save(tmp, format='JPEG', quality=90)
    replace_file(tmp, path)


def batch_preview(run, x, label, score, record, clip, args):
    directory = run/'previews'
    directory.mkdir(exist_ok=True)
    pixels = pixels_from_tensor(x)
    save_gif(directory/'train_input.gif', pixels, args.fps)
    save_strip(directory/'train_strip.jpg', pixels)
    raw = np.load(record['clips'][clip], allow_pickle=False)
    save_gif(directory/'train_decoded.gif', raw, args.fps)
    return dict(input='previews/train_input.gif', decoded='previews/train_decoded.gif',
                strip='previews/train_strip.jpg', label=int(label), score=float(score),
                prediction='FAKE' if score >= .5 else 'REAL', source=record['source'],
                filename=Path(record['path']).name, start=record['starts'][clip],
                input_mode=getattr(args, 'input_mode', 'rgb'),
                note='Score before this batch update; input after clip-consistent augmentation')


def gradcam(model, x):
    features = []
    # Eight flow steps collapse to one at layer4; layer2 keeps a coarser but useful time axis.
    layer = model.layer2 if x.shape[1] == 2 and hasattr(model, 'layer2') else model.layer4
    hook = layer.register_forward_hook(lambda _m, _i, output: features.append(output))
    try:
        with torch.enable_grad():
            logits = model(x.float()).flatten()
            score = float(logits.detach().sigmoid()[0])
            target = logits[0] if score >= .5 else -logits[0]
            gradient, = torch.autograd.grad(target, features[0])
            heat = (features[0]*gradient.mean((2, 3, 4), keepdim=True)).sum(1, keepdim=True).relu()
            heat = F.interpolate(heat, size=x.shape[2:], mode='trilinear', align_corners=False)[0, 0]
            heat = heat.detach().cpu().numpy()
            maximum = float(heat.max())
            return score, heat/max(maximum, 1e-12), maximum > 1e-12
    finally:
        hook.remove()


def diagnostics(model, dataset, args, device, run, epoch):
    was_training = model.training
    model.eval()
    results = []
    directory = run/'previews'
    directory.mkdir(exist_ok=True)
    # Fixed examples from both classes; never used as an optimization objective.
    indices = []
    for label in (0, 1):
        candidates = [i for i, (r, clip) in enumerate(dataset.items) if r['label'] == label and clip == 0]
        selected, sources = [], set()
        for index in candidates:
            source = dataset.items[index][0]['source']
            if source not in sources:
                selected.append(index)
                sources.add(source)
        selected += [i for i in candidates if i not in selected]
        indices += selected[:max(1, args.preview_count//2)]
    try:
        for slot, index in enumerate(indices):
            x, label, _ = dataset[index]
            record, clip = dataset.items[index]
            batch = x[None].to(device)
            score, heat, valid = gradcam(model, batch)
            raw = pixels_from_tensor(x)
            color = np.stack((heat, .3*heat, 1-heat), -1)*255
            overlay = np.clip(raw*.65 + color*.35, 0, 255).astype(np.uint8)
            prefix = f'epoch_{epoch:03}_{slot}'
            save_gif(directory/f'{prefix}_input.gif', raw, args.fps)
            save_gif(directory/f'{prefix}_cam.gif', overlay, args.fps)
            save_strip(directory/f'{prefix}_strip.jpg', raw)
            generator = torch.Generator().manual_seed(args.seed+slot)
            steps = x.shape[1]
            order = torch.randperm(steps, generator=generator).to(device)
            with torch.inference_mode():
                variants = dict(shuffled=batch[:, :, order], repeated=batch[:, :, :1].expand_as(batch))
                if is_motion(args):
                    variants.update(zero=motion_variant(batch, 'zero'), mean_field=motion_variant(batch, 'mean_field'))
                scores = {key: float(model(value).float().sigmoid().item()) for key, value in variants.items()}
                occlusion = []
                window = 2 if is_motion(args) else 4
                for begin in range(0, steps, window):
                    masked = batch.clone()
                    masked[:, :, begin:begin+window] = 0
                    occlusion.append(dict(begin=begin, end=min(begin+window, steps),
                                          score=float(model(masked).float().sigmoid().item())))
            results.append(dict(id=record['id'], source=record['source'], filename=Path(record['path']).name,
                                label=int(label), score=score, prediction='FAKE' if score >= .5 else 'REAL',
                                input=f'previews/{prefix}_input.gif', cam=f'previews/{prefix}_cam.gif',
                                strip=f'previews/{prefix}_strip.jpg', heat_valid=valid,
                                temporal_saliency=heat.mean((1, 2)).tolist(), variants=scores, occlusion=occlusion,
                                input_mode=getattr(args, 'input_mode', 'rgb')))
    finally:
        model.train(was_training)
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    report = dict(epoch=epoch, examples=results, attribution_layer='layer2' if is_motion(args) else 'layer4',
                  caveat='Grad-CAM is coarse, interpolated and relative per clip. '
                  'Perturbed inputs are out of distribution; changed scores do not prove artifact reasoning.')
    atomic_json(run/f'explanations_{epoch:03}.json', report)
    return report


def weight_probes(model):
    names = ('stem.0.weight', 'layer2.0.conv1.0.0.weight', 'layer4.0.conv1.0.0.weight', 'fc.weight')
    return {name: p.detach().flatten()[:4096].cpu().clone() for name, p in model.named_parameters() if name in names}


def weight_report(model, initial):
    current = weight_probes(model)
    return {name: dict(sampled_weights=len(value), rms=float(value.square().mean().sqrt()),
                       delta_rms=float((value-initial[name]).square().mean().sqrt()),
                       relative_change=float((value-initial[name]).norm()/initial[name].norm().clamp_min(1e-12)),
                       histogram=np.histogram(value.numpy(), bins=20)[0].tolist()) for name, value in current.items()}
