"""Isolated DFDC experiment: native face motion versus existing full-frame motion."""
import argparse
import cv2
import torch
from filelock import FileLock

from training.final_cohort import MOTION_FINAL, heldout_selection
from training.final_metrics import comparison, operating_point, read_json
from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.data import assert_disjoint
from training.spatial.engine import atomic_torch
from training.spatial.lineage import file_sha
from training.spatial.media import FrameCache
from training.spectral.faces import MODEL, SHA256
from training.temporal_model import build_temporal_model
from .dashboard import Monitor, start_server
from .engine import evaluate, freeze_face_coverage, predictions, train
from .face_region import VERSION, complete_pairs
from .flow import FLOW_PARAMS, FLOW_VERSION
from .media import prepare
from .paired_batch import verify_plans

PARENT = MOTION_FINAL.with_name('motion_paired_dfdc_v1')
RUN = PARENT.with_name('motion_face_dfdc_v1')


def initialize(run):
    weights = next((p for p in (MODEL, ROOT/'demo_weights'/MODEL.name)
                    if p.is_file() and file_sha(p) == SHA256), None)
    if weights is None:
        raise ValueError('Verified local YuNet weights missing; face experiment does not download weights')
    if (run/'config.json').exists():
        config = read_json(run/'config.json')
    else:
        parent = read_json(PARENT/'config.json')
        best = read_json(PARENT/'best.json')
        frozen = read_json(PARENT/'splits.json')
        if (not (PARENT/'final_report.json').is_file() or fingerprint(frozen['splits']) != frozen['fingerprint']
                or best['split_hash'] != frozen['fingerprint'] or parent['frame_view'] != 'full'):
            raise ValueError('Expected the completed, compatible full-frame paired experiment')
        cohort = read_json(PARENT/'motion_cohort.json')
        training = {r['id']: r for r in frozen['splits']['train']}
        eligible = {i: meta for i, meta, _ in cohort['cohorts']['train']}
        parent_plan = read_json(PARENT/'paired_plan.json')
        verify_plans(parent_plan, parent['paired_plan_hash'], eligible.keys(),
                     parent['epochs'], parent['videos_per_epoch'])
        plans = []
        for epoch, plan in enumerate(parent_plan['plans'][:3], 1):
            pairs = plan['pairs'][:150]
            plans.append(dict(epoch=epoch, pairs=pairs, ids=[i for pair in pairs for i in pair],
                              new_by_label={}))
        train_ids = sorted({i for p in plans for i in p['ids']})
        val_ids = {i for i, _, _ in cohort['cohorts']['val']}
        validation = heldout_selection([r for r in frozen['splits']['val'] if r['id'] in val_ids],
                                       parent['val_videos'], parent['seed'])
        test = heldout_selection(frozen['splits']['test'], parent['test_videos'], parent['seed'])
        splits = dict(train=[dict(training[i], metadata=eligible[i]) for i in train_ids],
                      val=complete_pairs([r for r in validation if r['source'] == 'DFDC']),
                      test=complete_pairs([r for r in test if r['source'] == 'DFDC']), ood=[])
        atomic_json(run/'splits.json', dict(splits=splits, fingerprint=fingerprint(splits)))
        plan_payload = dict(plans=plans)
        plan_hash = fingerprint(plan_payload)
        atomic_json(run/'paired_plan.json', dict(plan_payload, fingerprint=plan_hash))
        inherited = set(read_json(PARENT/'parent_best.json')['seen_ids'])
        for epoch in range(1, best['epoch']+1):
            inherited.update(read_json(PARENT/f'epoch_{epoch:03}_plan.json')['ids'])
        atomic_json(run/'previous_attempts.json', dict(ids=sorted(
            inherited | set(read_json(PARENT/'previous_attempts.json')['ids']) |
            set(read_json(PARENT/'coverage.json')['ids']))))
        parent_hash = file_sha(PARENT/'best.pt')
        atomic_torch(run/'initial.pt', torch.load(PARENT/'best.pt', map_location='cpu', weights_only=True))
        if file_sha(PARENT/'best.pt') != parent_hash:
            raise ValueError('Full-frame weights changed during initialization')
        config = dict(parent, recipe='temporal-motion-face-v1', frame_view='face',
                      face_version=VERSION, face_weights_sha256=SHA256, epochs=len(plans), patience=3,
                      videos_per_epoch=300, backbone_lr=5e-6, head_lr=3e-5, decode_workers=2,
                      parent_weights_sha256=parent_hash, paired_plan_hash=plan_hash,
                      previous_attempts_sha256=file_sha(run/'previous_attempts.json'),
                      **{'initial.pt_sha256': file_sha(run/'initial.pt')})
        atomic_json(run/'config.json', config)
    payload = read_json(run/'splits.json')
    if (config['face_version'], config['face_weights_sha256'], config['flow_version'],
        config['opencv_version'], config['flow_parameters']) != (VERSION, SHA256, FLOW_VERSION, cv2.__version__, FLOW_PARAMS):
        raise ValueError('Frozen face/flow preprocessing changed')
    if (fingerprint(payload['splits']) != payload['fingerprint']
            or file_sha(run/'initial.pt') != config['initial.pt_sha256']
            or file_sha(run/'previous_attempts.json') != config['previous_attempts_sha256']):
        raise ValueError('Frozen face experiment data changed')
    assert_disjoint(payload['splits'])
    verify_plans(read_json(run/'paired_plan.json'), config['paired_plan_hash'],
                 {r['id'] for r in payload['splits']['train']}, config['epochs'], config['videos_per_epoch'])
    return argparse.Namespace(**config, face_weights=str(weights)), payload


def checked_full_view(records, face_records, cache, tag, config, log, run, monitor):
    face_by_id = {r['id']: r for r in face_records}
    full = prepare([r for r in records if r['id'] in face_by_id], cache, tag, config.val_clips,
                   config, log, run, monitor.publish)
    if any(r['starts'] != face_by_id[r['id']]['starts'] for r in full):
        raise ValueError(f'{tag}: face and full-frame windows differ')
    return full


def report(run, payload, args, log, monitor):
    if (run/'final_report.json').exists():
        return
    candidates = payload['splits']
    cache = FrameCache(run, args.cache_gb, args.reserve_gb)
    prefix = run/'baseline'
    prefix.mkdir(exist_ok=True)
    full_cache = FrameCache(prefix, args.cache_gb/2, args.reserve_gb)
    full_args = argparse.Namespace(**dict(vars(args), frame_view='full'))
    device = torch.device(args.device)
    nets = [build_temporal_model(False, 'flow').to(device) for _ in range(2)]
    for net, name in zip(nets, ('initial.pt', 'best.pt')):
        net.load_state_dict(torch.load(run/name, map_location='cpu', weights_only=True))
    sides, thresholds, coverage = {}, {}, {}
    for split in ('val', 'test'):
        requested = candidates[split]
        face = prepare(requested, cache, split, args.val_clips, args, log, run, monitor.publish)
        freeze_face_coverage(run, split, requested, face, args.min_source_count if split == 'val' else 1)
        full = checked_full_view(requested, face, full_cache, split, full_args, log, prefix, monitor)
        coverage[split] = dict(requested_pairs=len(requested)//2, common_pairs=len(face)//2)
        sides[split] = dict(clean=[], stress=[])
        for name, net, rows, setup in zip(('baseline', 'candidate'), nets, (full, face), (full_args, args)):
            scored = []
            for mode in ('clean', 'stress'):
                scores, _ = evaluate(net, rows, setup, device, log, monitor, f'{split}/{name}/{mode}',
                                     stress=mode == 'stress')
                predictions(run/f'{split}_{name}_{mode}.csv', scores)
                sides[split][mode].append(scores)
                scored.append(scores)
            if split == 'val':
                thresholds[name] = operating_point(*scored)['threshold']
    result = comparison(run, 'test', sides['test'], thresholds, candidates['test'])
    final = dict(test=result, coverage=coverage, face_best_epoch=read_json(run/'best.json')['epoch'],
                 caveat='DFDC only; compares an existing full-frame model to a face-input candidate. '
                        'Both runs use identical supported videos and time starts. Not evidence for arbitrary generators.')
    atomic_json(run/'final_report.json', final)
    monitor.publish(final=final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--audit-only', action='store_true')
    modes.add_argument('--prepare-only', action='store_true', help='Check both views on the fixed validation pairs')
    parser.add_argument('--port', type=int, default=8767)
    args_cli = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ROOT/'cache/spatial_v2/training.lock'), timeout=0), FileLock(str(RUN/'run.lock'), timeout=0):
        log = logger_for(RUN)
        monitor = Monitor(RUN)
        server = None
        try:
            args, payload = initialize(RUN)
            log.info('FACE EXPERIMENT %s | requested %s', VERSION,
                     {k: len(v)//2 for k, v in payload['splits'].items()})
            monitor.publish(status='preparing', config={k: v for k, v in vars(args).items() if k != 'face_weights'})
            if args_cli.audit_only:
                monitor.publish(status='prepared', phase='ready')
                return
            if args_cli.prepare_only:
                cache = FrameCache(RUN, args.cache_gb, args.reserve_gb)
                validation = payload['splits']['val']
                face = prepare(validation, cache, 'val', args.val_clips, args, log, RUN, monitor.publish)
                freeze_face_coverage(RUN, 'val', validation, face, args.min_source_count)
                baseline = RUN/'baseline'
                baseline.mkdir(exist_ok=True)
                full_args = argparse.Namespace(**dict(vars(args), frame_view='full'))
                checked_full_view(validation, face, FrameCache(baseline, args.cache_gb/2, args.reserve_gb),
                                  'val', full_args, log, baseline, monitor)
                monitor.publish(status='prepared', phase='ready')
                return
            server, url = start_server(RUN, args_cli.port)
            log.info('DASHBOARD %s', url)
            model, _ = train(RUN, payload, payload['splits']['train'], payload['splits']['val'], args, log, monitor)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report(RUN, payload, args, log, monitor)
            monitor.publish(status='complete', phase='finished')
            log.info('FACE EXPERIMENT COMPLETE')
        except KeyboardInterrupt:
            monitor.publish(status='interrupted', phase='stopped')
            log.warning('Interrupted; repeat the same command to resume')
        except Exception as exc:
            monitor.publish(status='failed', error=str(exc))
            log.exception('Face experiment failed')
            raise
        finally:
            monitor.close()
            if server:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    main()
