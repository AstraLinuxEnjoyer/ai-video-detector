"""Bounded real-video integration run. Never modifies the production checkpoint."""
import argparse
import json
import shutil
from datetime import datetime

import torch
from filelock import FileLock

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.artifact_spatial.metadata import attach_dimensions
from training.artifact_spatial.sampling import MatchedRotation
from training.temporal_v2.dashboard import Monitor
from training.temporal_v2.engine import train
from training.temporal_v2.run import final_test


def main():
    source=ROOT/'cache/temporal_v2/runs/r2plus1d_v1'
    run=ROOT/'cache/temporal_v2/smoke'/datetime.now().strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    with FileLock(str(ROOT/'cache/spatial_v2/training.lock'),timeout=0):
        payload=json.loads((source/'splits.json').read_text(encoding='utf-8'))
        dimensions=json.loads((source/'dimensions.json').read_text())
        training=attach_dimensions([r for r in payload['splits']['train'] if r.get('kind')!='image_sequence'],dimensions)
        chosen=MatchedRotation(training).take(8)
        wanted=['KoNViD_1k_videos','VideoUFO','OpenAI_Sora','keling','GenVidBench:ms','GenVideo_ZeroScope']
        validation=[next(r for r in payload['splits']['val'] if r['source']==s) for s in wanted]
        test=[next(r for r in payload['splits']['test'] if r['source']==s) for s in wanted[:4]]
        config=json.loads((source/'config.json').read_text())
        config.update(smoke=True,epochs=1,videos_per_epoch=8,val_videos=6,test_videos=4,train_clips=1,val_clips=1,
                      min_source_count=1,preview_count=2,save_every=1,log_every=1)
        args=argparse.Namespace(**config)
        shutil.copy2(source/'initial.pt',run/'initial.pt')
        splits=dict(train=chosen,val=validation,test=test,ood=[])
        small=dict(splits=splits,fingerprint=fingerprint(splits))
        atomic_json(run/'config.json',config)
        atomic_json(run/'splits.json',small)
        log=logger_for(run)
        log.warning('SMOKE ONLY: tiny sample, not a quality benchmark')
        monitor=Monitor(run)
        try:
            monitor.publish(status='running',config=config)
            model,state=train(run,small,chosen,validation,args,log,monitor)
            assert state['complete'] and len(state['seen'])==8
            initial=torch.load(run/'initial.pt',weights_only=True,map_location='cpu')
            assert not torch.equal(initial['fc.weight'],model.fc.weight.detach().cpu())
            assert all(torch.isfinite(p).all() for p in model.parameters())
            for name,value in model.state_dict().items():
                if 'running_' in name or 'num_batches_tracked' in name:
                    torch.testing.assert_close(initial[name],value.cpu(),rtol=0,atol=0)
            del model
            torch.cuda.empty_cache()
            final_test(run,splits,args,log,monitor)
            monitor.publish(status='complete',phase='smoke_complete')
            print('TEMPORAL_GPU_SMOKE_OK',run)
            atomic_json(ROOT/'cache/temporal_v2/latest_smoke.json',dict(run_dir=str(run)))
        finally:monitor.close()


if __name__=='__main__':main()
