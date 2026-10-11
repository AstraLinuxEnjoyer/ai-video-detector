"""Bounded real-media integration check, separate from the actual experiment."""
import argparse
import json
import shutil
from datetime import datetime

import torch
from filelock import FileLock

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.artifact_spatial.engine import train
from training.artifact_spatial.metadata import attach_dimensions
from training.artifact_spatial.sampling import MatchedRotation


def main():
    parent = ROOT/'cache/artifact_spatial/runs/hybrid_v2'
    with FileLock(str(ROOT/'cache/spatial_v2/training.lock'), timeout=0):
        payload = json.loads((parent/'splits.json').read_text(encoding='utf-8'))
        index = json.loads((parent/'dimensions.json').read_text())
        rows = attach_dimensions(payload['splits']['train'], index)
        selected = MatchedRotation(rows).take(8)
        sources = ['DAVIS', 'KoNViD_1k_videos', 'VideoUFO', 'OpenAI_Sora', 'keling', 'GenVidBench:ms']
        val = [next(r for r in payload['splits']['val'] if r['source'] == s) for s in sources]
        splits = dict(train=selected, val=val, test=[], ood=[])
        run = ROOT/'cache/artifact_spatial/smoke'/('hybrid_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        run.mkdir(parents=True)
        for name in ('initial.pt', 'lineage.json', 'dimensions.json'):
            shutil.copy2(parent/name, run/name)
        config = json.loads((parent/'config.json').read_text())
        config.update(epochs=1, videos_per_epoch=8, val_videos=6, val_frames=2,
                      min_source_count=1, log_every=1, save_every=1)
        args = argparse.Namespace(**config)
        snapshot = dict(splits=splits, fingerprint=fingerprint(splits))
        atomic_json(run/'config.json', config)
        atomic_json(run/'splits.json', snapshot)
        log = logger_for(run)
        log.warning('SMOKE ONLY: 8 train / 6 validation. Not an accuracy benchmark.')
        model, state = train(run, snapshot, args, log)
        assert state['complete'] and len(state['seen']) == 8
        assert all(torch.isfinite(p).all() for p in model.parameters())
        initial = torch.load(run/'initial.pt', map_location='cpu', weights_only=True)
        assert not torch.equal(initial['classifier.1.weight'], model.classifier[1].weight.detach().cpu())
        for name, value in model.state_dict().items():
            if 'running_' in name or 'num_batches_tracked' in name:
                torch.testing.assert_close(value.cpu(), initial[name], rtol=0, atol=0)
        print('HYBRID_GPU_SMOKE_OK', run)


if __name__ == '__main__':
    main()
