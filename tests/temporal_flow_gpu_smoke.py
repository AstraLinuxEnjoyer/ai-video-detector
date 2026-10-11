"""One tiny, isolated epoch on existing real videos; never a quality benchmark."""
import argparse
import json
import shutil
from datetime import datetime

import torch
from filelock import FileLock

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.lineage import file_sha
from training.temporal_v2.dashboard import Monitor
from training.temporal_v2.engine import train
from training.temporal_v2.flow import rotation_for
from training.temporal_v2.run import initialize, final_test, choose_validation


def main():
    source = ROOT/'cache/temporal_v2/runs/motion_flow_v1'
    old = ROOT/'cache/temporal_v2/runs/r2plus1d_v1'
    protected = {name: file_sha(old/name) for name in ('best.pt', 'last.pt', 'checkpoint.pt')}
    run = ROOT/'cache/temporal_v2/smoke'/('flow_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True)
    with FileLock(str(ROOT/'cache/spatial_v2/training.lock'), timeout=0):
        args, payload, training, eligible = initialize(source, None, {'input_mode': 'flow'}, logger_for(source))
        config = dict(vars(args), smoke=True, epochs=1, videos_per_epoch=8, val_videos=8,
                      test_videos=4, train_clips=1, val_clips=1, min_source_count=1,
                      preview_count=2, save_every=1, log_every=1)
        args = argparse.Namespace(**config)
        chosen = rotation_for(training, args).take(8)
        validation = choose_validation(eligible['val'], args)
        # Use disjoint held-out IDs with matching native FPS, retaining the original split labels.
        from training.spatial.media import probe
        test = []
        for label in (0, 1):
            for record in eligible['test']:
                if record['label'] != label:
                    continue
                meta = probe(record['path'])
                if meta['fps'] >= args.fps and meta['duration'] >= args.clip_length/args.fps:
                    test.append(record)
                    break
        splits = dict(train=chosen, val=validation, test=test, ood=[])
        small = dict(splits=splits, fingerprint=fingerprint(splits))
        shutil.copy2(source/'initial.pt', run/'initial.pt')
        atomic_json(run/'config.json', config)
        atomic_json(run/'splits.json', small)
        log = logger_for(run)
        log.warning('MOTION SMOKE ONLY: 8 training videos, not a quality benchmark')
        monitor = Monitor(run)
        try:
            monitor.publish(status='running', config=config)
            model, state = train(run, small, chosen, validation, args, log, monitor)
            assert state['complete'] and len(state['seen']) == 8
            assert state['schema'] == 'temporal-motion-v1'
            initial = torch.load(run/'initial.pt', weights_only=True, map_location='cpu')
            assert not torch.equal(initial['fc.weight'], model.fc.weight.detach().cpu())
            assert model.stem[0].in_channels == 2
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert not torch.equal(initial['stem.1.running_mean'], model.stem[1].running_mean.cpu())
            del model, initial
            torch.cuda.empty_cache()
            final_test(run, splits, args, log, monitor)
            monitor.publish(status='complete', phase='smoke_complete')
            print('MOTION_GPU_SMOKE_OK', run)
            atomic_json(ROOT/'cache/temporal_v2/latest_flow_smoke.json', dict(run_dir=str(run)))
        finally:
            monitor.close()
            assert protected == {name: file_sha(old/name) for name in protected}


if __name__ == '__main__':
    main()
