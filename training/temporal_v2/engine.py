"""Resumable clip learning, source-aware selection and diagnostic snapshots."""
import csv
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from training.temporal_model import build_temporal_model
from training.spatial.common import atomic_json, event, fingerprint
from training.spatial.engine import atomic_torch, loader, optimizer_batch, save_state, seed_all
from training.spatial.media import FrameCache
from training.spatial.metrics import grouped_report, metrics, bootstrap_interval
from training.artifact_spatial.objective import selection_score
from training.artifact_spatial.sampling import MatchedRotation
from .media import Clips, clips_for, prepare
from .flow import is_motion, motion_bucket, motion_variant, rotation_for
from .explain import batch_preview, diagnostics, weight_probes, weight_report


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def predictions(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, ['id', 'path', 'group', 'source', 'label', 'score', 'clip_scores'])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, records, args, device, log, monitor, tag, stress=False, variant=None):
    model.eval()
    dataset = clips_for(records, args, stress=stress)
    batches = loader(dataset, args.batch_size, args.workers)
    scores, clip_y, clip_p = defaultdict(list), [], []
    with torch.inference_mode():
        for step, (x, y, indices) in enumerate(tqdm(batches, desc=tag), 1):
            if variant:
                x = motion_variant(x, variant, args.seed)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                p = model(x.to(device)).flatten().float().sigmoid().cpu()
            if not torch.isfinite(p).all():
                raise RuntimeError('Nonfinite validation output')
            clip_y.extend(y.tolist())
            clip_p.extend(p.tolist())
            for index, value in zip(indices.tolist(), p.tolist()):
                record, _ = dataset.items[index]
                scores[record['id']].append(value)
            if step % args.log_every == 0 or step == len(batches):
                monitor.publish(phase=tag, done=step, total=len(batches))
    rows = [dict(id=r['id'], path=r['path'], group=r['group'], source=r['source'], label=r['label'],
                 score=float(np.mean(scores[r['id']])), clip_scores=scores[r['id']]) for r in records]
    report = grouped_report(rows)
    report['clips'] = metrics(clip_y, clip_p)
    by_id = {r['id']: r for r in records}
    fps_groups = defaultdict(list)
    for row in rows:
        native = by_id[row['id']].get('metadata', {}).get('fps')
        key = 'unknown' if native is None else 'below_16' if native < 16 else '16_and_above'
        fps_groups[key].append(row)
    report['by_native_fps'] = {key: grouped_report(values) for key, values in fps_groups.items()}
    if is_motion(args):
        known = [row for row in rows if by_id[row['id']].get('metadata', {}).get('fps') is not None]
        if known:
            report['metadata_fps_baseline'] = metrics([r['label'] for r in known],
                       [float(by_id[r['id']]['metadata']['fps'] < 16) for r in known])
        shared = {key: value['balanced_accuracy'] for key, value in report['by_native_fps'].items()
                  if key != 'unknown' and {r['label'] for r in fps_groups[key]} == {0, 1}}
        report['fps_stratification'] = dict(shared_bins=shared,
                    balanced_accuracy=float(np.mean(list(shared.values()))) if shared else None,
                    unsupported_bins=[key for key in fps_groups if key not in shared],
                    caveat='Coarse native-FPS bins; not exact codec/FPS matching or a physics test')
        log.info('%s FPS AUDIT metadata_only_accuracy=%s stratified_balanced_accuracy=%s shared_bins=%s',
                 tag, report.get('metadata_fps_baseline', {}).get('accuracy'),
                 report['fps_stratification']['balanced_accuracy'], shared)
    report['confident_errors'] = sum(abs(r['score']-r['label']) >= .95 for r in rows)
    report['group_bootstrap_95ci'] = bootstrap_interval(rows)
    report['stress_recipe'] = (('flow field blur5 + velocity quantization1/1024; NOT a video recompression test'
                               if is_motion(args) else 'spatial resize + JPEG65 + hold every fourth frame') if stress else None)
    log.info('%s videos=%d accuracy=%.4f balanced=%s FPR=%s recall_fake=%s clip_accuracy=%.4f',
             tag, len(rows), report['accuracy'], report['balanced_accuracy'], report['false_positive_rate'],
             report['recall_fake'], report['clips']['accuracy'])
    for source, m in report['by_source'].items():
        log.info('%s SOURCE %s n=%d TP=%d TN=%d FP=%d FN=%d', tag, source, m['n'], m['tp'], m['tn'], m['fp'], m['fn'])
    return rows, report


def train(run, payload, train_records, chosen_val, args, log, monitor):
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; fix PyTorch or explicitly choose --device cpu in a new run')
    model = (build_temporal_model(False, 'flow') if is_motion(args) else build_temporal_model(False)).to(device)
    initial_weights = torch.load(run/'initial.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(initial_weights)
    initial = weight_probes(model)
    del initial_weights
    backbone = [p for name, p in model.named_parameters() if not name.startswith('fc.')
                and not (is_motion(args) and name.startswith('stem.'))]
    groups = [dict(params=backbone, lr=args.backbone_lr),
              dict(params=model.fc.parameters(), lr=args.head_lr)]
    if is_motion(args):
        groups.append(dict(params=model.stem.parameters(), lr=args.head_lr))
        log.info('MOTION OPTIMIZER backbone_lr=%.2g classifier_lr=%.2g motion_stem_lr=%.2g; '
                 'BatchNorm statistics adapt to velocity fields', args.backbone_lr, args.head_lr, args.head_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=.5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=1024)
    state = dict(schema='temporal-motion-v1' if is_motion(args) else 'temporal-v2', split_hash=payload['fingerprint'], epoch=0, offset=0,
                 plan=None, rotation=None, seen=[], best=-1., stale=0, complete=False,
                 loss_sum=0., correct=0, count=0, retries=0, reserve_gb=args.reserve_gb)
    if is_motion(args):
        state['cohort_hash'] = getattr(args, 'motion_cohort_hash',
                                      fingerprint(sorted((r['id'], motion_bucket(r)) for r in train_records)))
    if (run/'checkpoint.pt').exists():
        saved = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False)
        if saved['split_hash'] != payload['fingerprint'] or saved['schema'] != state['schema']:
            raise ValueError('Checkpoint and frozen partitions disagree')
        if is_motion(args) and saved.get('cohort_hash') != state['cohort_hash']:
            raise ValueError('Checkpoint and frozen motion cohort disagree')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        state.update({k: saved[k] for k in state})
        torch.set_rng_state(saved['torch_rng'])
        import random
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        if device.type == 'cuda' and saved.get('cuda_rng'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        log.info('RESUME epoch=%d clip_offset=%d seen=%d complete=%s', state['epoch']+1, state['offset'], len(state['seen']), state['complete'])
    else:
        save_state(run, model, optimizer, scheduler, scaler, state)
    cache = FrameCache(run, args.cache_gb, args.reserve_gb)
    history_path = run/'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if state['complete']:
        return model, state
    validation = prepare(chosen_val, cache, 'val', args.val_clips, args, log, run, monitor.publish)
    if not (run/'initial.json').exists():
        if state['epoch'] or state['offset']:
            raise RuntimeError('Missing initial report after training started')
        _, baseline = evaluate(model, validation, args, device, log, monitor, 'initial/val')
        explanation = diagnostics(model, clips_for(validation, args), args, device, run, 0)
        atomic_json(run/'initial.json', baseline)
        monitor.publish(baseline=baseline, explanation=explanation, weights=weight_report(model, initial))
    records_by_id = {r['id']: r for r in train_records}
    rotation = rotation_for(train_records, args, state=state['rotation'])
    while state['epoch'] < args.epochs and not state['complete']:
        epoch = state['epoch']
        monitor.publish(epoch=epoch+1, max_epochs=args.epochs, phase='prepare', status='running')
        if state['plan'] is None:
            # The plan and the sampler cursor commit together, before any optimization.
            plan = rotation.take(args.videos_per_epoch)
            state.update(plan=[r['id'] for r in plan], rotation=rotation.state, offset=0,
                         loss_sum=0., correct=0, count=0, retries=0)
            atomic_json(run/f'epoch_{epoch+1:03}_plan.json', dict(ids=state['plan'],
                        sources=dict(Counter(r['source'] for r in plan)), labels=dict(Counter(r['label'] for r in plan))))
            save_state(run, model, optimizer, scheduler, scaler, state)
        plan = [records_by_id[i] for i in state['plan']]
        if epoch:
            cache.clear_train(epoch-1)
        training = prepare(plan, cache, f'train_{epoch}', args.train_clips, args, log, run, monitor.publish)
        dataset = clips_for(training, args, seed=args.seed+epoch)
        batches = loader(dataset, args.batch_size, args.workers, state['offset'])
        model.train()
        if not is_motion(args):
            freeze_bn(model)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        total = (len(dataset)+args.batch_size-1)//args.batch_size
        log.info('EPOCH %d/%d videos=%d clips=%d sources=%s', epoch+1, args.epochs, len(plan), len(dataset), dict(Counter(r['source'] for r in plan)))
        loop = tqdm(batches, total=total, initial=state['offset']//args.batch_size, desc=f'TEMPORAL {epoch+1}')
        end = time.perf_counter()
        for x, labels, indices in loop:
            wait_seconds = time.perf_counter()-end
            begin = time.perf_counter()
            before = model.fc.weight.detach().clone()
            x_device = x.to(device, non_blocking=True)
            target = (labels.to(device).float()*.98+.01).view(-1, 1)
            loss, logits, norm, retries = optimizer_batch(model, x_device, target, optimizer, scaler,
                    log, run, dict(epoch=epoch+1, offset=state['offset']))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            compute_seconds = time.perf_counter()-begin
            p = logits.float().sigmoid().flatten().cpu()
            delta = float((model.fc.weight.detach()-before).square().mean().sqrt())
            state['offset'] += len(labels)
            state['count'] += len(labels)
            state['loss_sum'] += float(loss)*len(labels)
            state['correct'] += int(((p >= .5) == labels.bool()).sum())
            state['retries'] += retries
            seen = set(state['seen'])
            seen.update(dataset.items[i][0]['id'] for i in indices.tolist())
            state['seen'] = sorted(seen)
            step = (state['offset']+args.batch_size-1)//args.batch_size
            loop.set_postfix(loss=f'{float(loss):.4f}', acc=f"{state['correct']/state['count']:.4f}")
            if step % args.log_every == 0 or step == 1 or state['offset'] == len(dataset):
                record, clip = dataset.items[int(indices[0])]
                preview = batch_preview(run, x[0], labels[0], p[0], record, clip, args)
                values = dict(phase='train', done=step, total=total, epoch=epoch+1,
                              loss=float(loss), train_loss=state['loss_sum']/state['count'],
                              train_accuracy=state['correct']/state['count'], gradient_norm=float(norm),
                              head_update_rms=delta, weights=weight_report(model, initial),
                              lr=[g['lr'] for g in optimizer.param_groups], data_wait_ms=wait_seconds*1000,
                              compute_ms=compute_seconds*1000, clips_per_second=len(labels)/max(wait_seconds+compute_seconds, 1e-9),
                              gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0.,
                              seen=len(seen), train_pool=len(train_records), sample=preview, retries=state['retries'])
                monitor.record(**values)
                log.info('BATCH %d/%d loss=%.5f acc=%.4f grad=%.3f head_delta=%.3g wait=%.3fs compute=%.3fs seen=%d',
                         step, total, float(loss), values['train_accuracy'], float(norm), delta, wait_seconds, compute_seconds, len(seen))
            if step % args.save_every == 0 or state['offset'] == len(dataset):
                save_state(run, model, optimizer, scheduler, scaler, state)
                monitor.publish(checkpoint_offset=state['offset'])
            end = time.perf_counter()
        rows, clean = evaluate(model, validation, args, device, log, monitor, 'validation/clean')
        stress_rows, stress = evaluate(model, validation, args, device, log, monitor, 'validation/stress', True)
        selection = selection_score(clean, stress, args.min_source_count)
        report = dict(epoch=epoch+1, train_accuracy=state['correct']/state['count'],
                      train_loss=state['loss_sum']/state['count'], clean=clean, stress=stress,
                      selection=selection, unique_train_seen=len(state['seen']), retries=state['retries'])
        if is_motion(args):
            report['motion_controls'] = motion_controls(model, validation, rows, args, device, log, monitor)
        if selection['score'] > state['best']:
            state.update(best=selection['score'], stale=0)
            atomic_torch(run/'best.pt', model.state_dict())
            atomic_json(run/'best.json', dict(epoch=epoch+1, selection=selection, threshold=.5,
                        calibrated=False, split_hash=payload['fingerprint'], aggregation='mean_clips',
                        recipe='temporal-motion-v1' if is_motion(args) else 'temporal-v2'))
        else:
            state['stale'] += 1
        scheduler.step(selection['score'])
        report['stale'] = state['stale']
        atomic_json(run/f'epoch_{epoch+1:03}.json', report)
        predictions(run/f'epoch_{epoch+1:03}_clean.csv', rows)
        predictions(run/f'epoch_{epoch+1:03}_stress.csv', stress_rows)
        explanation = diagnostics(model, clips_for(validation, args), args, device, run, epoch+1)
        history = [r for r in history if r['epoch'] != epoch+1]+[report]
        atomic_json(history_path, history)
        monitor.publish(validation=report, explanation=explanation,
                        best=json.loads((run/'best.json').read_text()), weights=weight_report(model, initial))
        state.update(epoch=epoch+1, plan=None, offset=0,
                     complete=state['stale'] >= args.patience or epoch+1 >= args.epochs)
        atomic_torch(run/'last.pt', model.state_dict())
        save_state(run, model, optimizer, scheduler, scaler, state)
        atomic_json(run/'coverage.json', dict(unique_seen=len(state['seen']), eligible=len(train_records), ids=state['seen']))
        log.info('EPOCH DONE %d selection=%.5f best=%.5f stale=%d/%d train=%.4f seen=%d',
                 epoch+1, selection['score'], state['best'], state['stale'], args.patience, report['train_accuracy'], len(state['seen']))
        cache.clear_train(epoch)
    log.info('TRAIN COMPLETE reason=%s', 'early_stopping' if state['stale']>=args.patience else 'epoch_limit')
    return model, state


def motion_controls(model, records, clean_rows, args, device, log, monitor):
    baseline = {r['id']: r for r in clean_rows}
    reports = {}
    for variant in ('zero', 'mean_field', 'shuffle'):
        rows, report = evaluate(model, records, args, device, log, monitor, 'motion_control/'+variant,
                                variant=variant)
        report.update(decisions_changed=sum((r['score'] >= .5) != (baseline[r['id']]['score'] >= .5) for r in rows),
                      mean_absolute_score_change=float(np.mean([abs(r['score']-baseline[r['id']]['score']) for r in rows])),
                      mean_score=float(np.mean([r['score'] for r in rows])))
        log.info('MOTION CONTROL %s changed=%d/%d mean_abs_delta=%.4f score=%.4f', variant,
                 report['decisions_changed'], len(rows), report['mean_absolute_score_change'], report['mean_score'])
        reports[variant] = report
    return dict(variants=reports, caveat='Perturbations of estimated flow, not physical motion labels. '
                'Sensitivity is not proof of physics understanding. Zero input may be out of distribution.')
