import argparse
import copy
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import torch

from training.spatial.common import atomic_json, fingerprint
from training.spatial.engine import atomic_torch
from training.spatial.media import FrameCache
from training.temporal_v2 import engine, media
from training.temporal_v2.dashboard import start_server
from training.temporal_v2.explain import gradcam
from training.temporal_v2.run import DEFAULTS, evaluation_coverage, heldout_evaluation


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(torch.nn.Conv3d(3, 2, 1), torch.nn.BatchNorm3d(2), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Conv3d(2, 2, 1), torch.nn.LeakyReLU(.1))
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(self.layer4(self.stem(x)).mean((2, 3, 4)))


class Monitor:
    def publish(self, **values):
        pass

    record = publish


def fixture(root):
    args = argparse.Namespace(**dict(DEFAULTS, epochs=1, videos_per_epoch=8, batch_size=2,
                train_clips=1, val_clips=1, image_size=32, decode_size=32, clip_length=4,
                workers=0, min_source_count=1, preview_count=2, log_every=50, save_every=1,
                reserve_gb=0., device='cpu'))
    groups = {}
    for split in ('train', 'val'):
        rows = []
        for label in (0, 1):
            for i in range(4 if split == 'train' else 1):
                name = f'{split}_{label}_{i}'
                path = root/f'{name}.npy'
                np.save(path, np.full((4, 32, 32, 3), 50+label*100+i, np.uint8))
                rows.append(dict(id=name, path=str(path), group=name, label=label, source=f'source{label}',
                                 metadata=dict(width=320, height=240), clips=[str(path)], starts=[0.]))
        groups[split] = rows
    payload = dict(splits=groups, fingerprint=fingerprint(groups))
    return args, payload


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_starts_are_fixed_or_seeded(self):
        self.assertEqual(media.starts(5,16,16,3,42,False),media.starts(5,16,16,3,99,False))
        self.assertEqual(media.starts(5,16,16,3,42,True),media.starts(5,16,16,3,42,True))
        self.assertNotEqual(media.starts(5,16,16,3,42,True),media.starts(5,16,16,3,43,True))
        with self.assertRaises(ValueError):
            media.starts(.1,16,16,3,42,False)

    def test_clip_consistent_augmentation(self):
        frame = np.random.default_rng(42).integers(0,256,(128,128,3),dtype=np.uint8)
        pixels = np.repeat(frame[None],16,0)
        for stress in (False,True):
            a = media.tensor_from_pixels(pixels,112,42,stress)
            b = media.tensor_from_pixels(pixels,112,42,stress)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(a[:,0],a[:,15],rtol=0,atol=0)
            self.assertEqual(a.shape,(3,16,112,112))

    def test_stress_frame_holding_preserves_order(self):
        pixels=np.stack([np.full((128,128,3),i*10,np.uint8) for i in range(16)])
        x=media.tensor_from_pixels(pixels,112,stress=True)
        for start in range(0,16,4):
            for index in range(start+1,start+4):
                torch.testing.assert_close(x[:,start],x[:,index],rtol=0,atol=0)
        self.assertTrue(torch.all(x[:,4] > x[:,0]))

    def test_failed_decode_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            video=root/'bad.mp4'
            video.write_bytes(b'bad')
            r=dict(id='bad',path=str(video),size=3,mtime=video.stat().st_mtime_ns,label=0,source='real')
            args=argparse.Namespace(**DEFAULTS)
            cache=FrameCache(root,1,0)
            with patch.object(media,'probe',side_effect=ValueError('broken')):
                with self.assertRaisesRegex(RuntimeError,'no replacement'):
                    media.prepare([r],cache,'val',1,args,logging.getLogger('test'),root)
            report=json.loads((root/'val_decode.json').read_text())
            self.assertEqual(report['usable'],0)
            self.assertEqual(len(report['errors']),1)

    def test_short_video_is_explicit_abstention_only_on_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = []
            for name, label in (('short', 1), ('normal', 0)):
                video = root/f'{name}.mp4'
                video.write_bytes(b'video')
                rows.append(dict(id=name, path=str(video), size=5, mtime=video.stat().st_mtime_ns,
                                 group=name, label=label, source=f'source{label}'))
            args = argparse.Namespace(**DEFAULTS)
            def metadata(path):
                return dict(duration=.8 if path.stem == 'short' else 2., fps=30)
            pixels = np.zeros((16, 128, 128, 3), np.uint8)
            with patch.object(media, 'probe', side_effect=metadata), patch.object(media, 'decode_clip', return_value=pixels) as decode:
                prepared = media.prepare(rows, FrameCache(root, 1, 0), 'ood', 1, args,
                                         logging.getLogger('test'), root, allow_short=True)
                self.assertEqual([r['id'] for r in prepared], ['normal'])
                self.assertEqual(decode.call_count, 1)
                self.assertEqual(decode.call_args.args[0].stem, 'normal')
                with self.assertRaisesRegex(RuntimeError, 'no replacement'):
                    media.prepare(rows[:1], FrameCache(root, 1, 0), 'train_0', 1, args,
                                  logging.getLogger('test'), root)
                with self.assertRaisesRegex(RuntimeError, 'no replacement'):
                    media.prepare(rows[:1], FrameCache(root, 1, 0), 'val', 1, args,
                                  logging.getLogger('test'), root)
            for tag in ('val', 'train_0'):
                with self.assertRaisesRegex(ValueError, 'held-out'):
                    media.prepare([], FrameCache(root, 1, 0), tag, 1, args,
                                  logging.getLogger('test'), root, allow_short=True)
            report = json.loads((root/'ood_decode.json').read_text())
            self.assertEqual(report['requested'], 2)
            self.assertEqual(report['usable'], 1)
            self.assertEqual(report['unsupported'][0]['reason'], 'clip_too_short')
            self.assertEqual(report['unsupported'][0]['duration'], .8)

    def test_other_decode_errors_remain_fatal_in_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root/'bad.mp4'
            video.write_bytes(b'bad')
            row = dict(id='bad', path=str(video), size=3, mtime=video.stat().st_mtime_ns,
                       label=1, source='fake')
            with patch.object(media, 'probe', side_effect=ValueError('broken')):
                with self.assertRaisesRegex(RuntimeError, 'no replacement'):
                    media.prepare([row], FrameCache(root, 1, 0), 'ood', 1,
                                  argparse.Namespace(**DEFAULTS), logging.getLogger('test'), root,
                                  allow_short=True)
            self.assertEqual(json.loads((root/'ood_decode.json').read_text())['unsupported'], [])

    def test_coverage_accounts_for_abstentions_in_original_denominator(self):
        requested = [dict(id='real', source='camera', label=0),
                     dict(id='fake', source='generator', label=1),
                     dict(id='short', source='generator', label=1)]
        rows = [dict(requested[0], score=.1), dict(requested[1], score=.2)]
        report = evaluation_coverage(requested, rows, [requested[2]])
        self.assertEqual(report['requested'], 3)
        self.assertEqual(report['evaluated'], 2)
        self.assertEqual(report['unsupported'], 1)
        self.assertEqual(report['accuracy_all_requested_bounds'], [1/3, 2/3])
        self.assertEqual(report['by_source']['generator']['coverage'], .5)
        self.assertEqual(report['by_class']['1']['accuracy_all_requested_bounds'], [0., .5])
        with self.assertRaisesRegex(ValueError, 'every requested ID'):
            evaluation_coverage(requested, rows, [])
        with self.assertRaisesRegex(ValueError, 'every requested ID'):
            evaluation_coverage(requested, rows, [requested[0], requested[2]])

    def test_all_short_external_videos_have_no_fabricated_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root/'short.mp4'
            video.write_bytes(b'video')
            row = dict(id='short', path=str(video), size=5, mtime=video.stat().st_mtime_ns,
                       group='short', label=1, source='fake')
            args = argparse.Namespace(**dict(DEFAULTS, device='cpu'))
            with patch.object(media, 'probe', return_value=dict(duration=.8, fps=30)), \
                    patch('training.temporal_v2.run.evaluate') as evaluate:
                rows, report = heldout_evaluation(None, [row], FrameCache(root, 1, 0),
                                                 'external', args, logging.getLogger('test'), root, Monitor())
                evaluate.assert_not_called()
            self.assertEqual(rows, [])
            self.assertIsNone(report['accuracy'])
            self.assertEqual(report['coverage']['coverage'], 0.)
            self.assertEqual(report['coverage']['accuracy_all_requested_bounds'], [0., 1.])

    def test_same_video_seek_recovery_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            video=root/'video.mp4'
            video.write_bytes(b'video')
            r=dict(id='sample',path=str(video),size=5,mtime=video.stat().st_mtime_ns,label=1,source='fake')
            args=argparse.Namespace(**DEFAULTS)
            pixels=np.zeros((16,128,128,3),np.uint8)
            with patch.object(media,'probe',return_value=dict(duration=5.,fps=24)),patch.object(media,'decode_clip',side_effect=[RuntimeError('short tail'),pixels]) as decode:
                rows=media.prepare([r],FrameCache(root,1,0),'val',1,args,logging.getLogger('test'),root)
            self.assertEqual(decode.call_args_list[0].args[0],decode.call_args_list[1].args[0])
            self.assertEqual(rows[0]['starts'],[1.75])
            self.assertEqual(len(rows[0]['seek_recoveries']),1)
            report=json.loads((root/'val_decode.json').read_text())
            self.assertEqual(report['usable'],1)
            self.assertEqual(len(report['recovered_seeks']),1)

    def test_gradcam_does_not_update_weights(self):
        model=Tiny().eval()
        old=copy.deepcopy(model.state_dict())
        score,heat,_=gradcam(model,torch.rand(1,3,4,32,32))
        self.assertTrue(0 <= score <= 1)
        self.assertEqual(heat.shape,(4,32,32))
        for name,value in model.state_dict().items():
            torch.testing.assert_close(old[name],value,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_dashboard_has_no_arbitrary_file_access(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            atomic_json(root/'live.json',dict(status='test'))
            (root/'secret.txt').write_text('private')
            server,url=start_server(root,0)
            try:
                self.assertEqual(json.load(urlopen(url+'/api/state'))['status'],'test')
                for route in ('/secret.txt','/previews/%2e%2e/secret.txt','/previews/%2e%2e/live.json'):
                    with self.assertRaises(HTTPError):urlopen(url+route)
            finally:
                server.shutdown()
                server.server_close()

    def test_durable_checkpoint_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            args,payload=fixture(root)
            torch.manual_seed(123)
            initial=Tiny().state_dict()
            a,b=root/'a',root/'b'
            for run in (a,b):
                run.mkdir()
                atomic_torch(run/'initial.pt',initial)
            def prepare(rows,*_args,**_kwargs):return rows
            original=engine.optimizer_batch
            calls=0
            def interrupt(*a,**kw):
                nonlocal calls
                calls+=1
                if calls==2:raise KeyboardInterrupt()
                return original(*a,**kw)
            with patch.object(engine,'build_temporal_model',lambda *_:Tiny()),patch.object(engine,'prepare',prepare):
                engine.train(a,payload,payload['splits']['train'],payload['splits']['val'],args,logging.getLogger('test'),Monitor())
                with patch.object(engine,'optimizer_batch',interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        engine.train(b,payload,payload['splits']['train'],payload['splits']['val'],args,logging.getLogger('test'),Monitor())
                checkpoint=torch.load(b/'checkpoint.pt',weights_only=False)
                self.assertEqual(checkpoint['offset'],2)
                engine.train(b,payload,payload['splits']['train'],payload['splits']['val'],args,logging.getLogger('test'),Monitor())
            x=torch.load(a/'checkpoint.pt',weights_only=False)
            y=torch.load(b/'checkpoint.pt',weights_only=False)
            self.assertTrue(y['complete'])
            self.assertEqual(x['seen'],y['seen'])
            for name,value in x['model'].items():
                torch.testing.assert_close(value,y['model'][name],rtol=0,atol=0)
            self.assertFalse(torch.equal(initial['fc.weight'],y['model']['fc.weight']))
            torch.testing.assert_close(initial['stem.1.running_mean'],y['model']['stem.1.running_mean'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
