"""Regression coverage for decoder drift and truthful spectral readiness."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from demo import app, runtime, spectral
from demo.check_runtime import check_runtime

EXPECTED = dict(ffmpeg='ffmpeg version 8.1.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers',
                numpy='2.5.0', opencv='5.0.0')


class SpectralRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model, self.face = self.root/'model.json', self.root/'face.onnx'
        self.face.write_bytes(b'known fixture face weights')
        self.checksum = hashlib.sha256(self.face.read_bytes()).hexdigest()
        self.saved = dict(config=dict(recipe='spectral-face-v3', versions=EXPECTED,
                                     yunet_sha256=self.checksum, frames=3), models=dict(full_face={}))
        self.save()
        for item in (patch.object(spectral, 'assets', return_value=(self.model, self.face)),
                     patch.object(spectral, 'SHA256', self.checksum),
                     patch.object(spectral, 'configure_tools'),
                     patch.object(spectral, 'versions', return_value=copy.deepcopy(EXPECTED)),
                     patch.object(spectral.shutil, 'which', side_effect=lambda name: 'local/'+name)):
            item.start()
            self.addCleanup(item.stop)

    def save(self):
        self.model.write_text(json.dumps(self.saved), encoding='utf-8')

    def test_matching_runtime_loads(self):
        self.assertEqual(spectral.spectral_status()['status'], 'ready')
        self.assertEqual(spectral.SpectralDetector().info['recipe'], 'spectral-face-v3')

    def test_newer_ffmpeg_is_rejected_without_rewriting_weights(self):
        before = self.model.read_bytes()
        actual = dict(EXPECTED, ffmpeg='ffmpeg version 9.0.2-full_build-www.gyan.dev')
        with patch.object(spectral, 'versions', return_value=actual):
            state = spectral.spectral_status()
            self.assertEqual(state['status'], 'unavailable')
            self.assertIn('8.1.1', state['message'])
            self.assertIn('9.0.2', state['message'])
            with self.assertRaisesRegex(ValueError, 'FFmpeg: expected'):
                spectral.SpectralDetector()
        self.assertEqual(self.model.read_bytes(), before)

    def test_numpy_drift_is_rejected(self):
        with patch.object(spectral, 'versions', return_value=dict(EXPECTED, numpy='2.6.0')):
            self.assertIn('NumPy: expected', spectral.spectral_status()['message'])

    def test_opencv_drift_is_rejected(self):
        with patch.object(spectral, 'versions', return_value=dict(EXPECTED, opencv='4.12.0')):
            self.assertIn('OpenCV: expected', spectral.spectral_status()['message'])

    def test_corrupt_yunet_is_rejected(self):
        self.face.write_bytes(b'damaged')
        self.assertIn('YuNet file checksum', spectral.spectral_status()['message'])

    def test_wrong_yunet_metadata_is_rejected(self):
        self.saved['config']['yunet_sha256'] = '0'*64
        self.save()
        self.assertIn('YuNet metadata checksum', spectral.spectral_status()['message'])

    def test_missing_face_has_actionable_message(self):
        self.face.unlink()
        state = spectral.spectral_status()
        self.assertEqual(state['status'], 'unavailable')
        self.assertIn('YuNet weights missing', state['message'])

    def test_invalid_model_json_is_reported(self):
        self.model.write_text('invalid json', encoding='utf-8')
        self.assertIn('Cannot validate spectral assets', spectral.spectral_status()['message'])

    def test_recipe_contract_is_preserved(self):
        self.saved['config']['recipe'] = 'another-recipe'
        self.saved['config']['frames'] = 9
        self.save()
        state = spectral.spectral_status()
        self.assertEqual(state['status'], 'unavailable')
        self.assertIn('Spectral recipe', state['message'])
        self.assertIn('exactly 3 frames', state['message'])

    def test_missing_ffprobe_is_reported(self):
        with patch.object(spectral.shutil, 'which', side_effect=lambda name: None if name=='ffprobe' else 'ffmpeg'):
            self.assertIn('FFprobe missing', spectral.spectral_status()['message'])

    def test_missing_ffmpeg_is_reported(self):
        with patch.object(spectral, 'versions', side_effect=FileNotFoundError('ffmpeg')):
            self.assertIn('Cannot inspect FFmpeg/runtime', spectral.spectral_status()['message'])


class LocalDecoderTests(unittest.TestCase):
    def test_complete_pair_precedes_system_path_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root/'tools'
            tools.mkdir()
            for name in ('ffmpeg', 'ffprobe'):
                (tools/(name+('.exe' if os.name=='nt' else ''))).touch()
            with patch.object(runtime, 'ROOT', root), patch.dict(os.environ, {'PATH':'system-tools'}):
                self.assertEqual(runtime.configure_tools(), tools)
                first = os.environ['PATH']
                runtime.configure_tools()
                self.assertEqual(os.environ['PATH'], first)
                self.assertEqual(first.split(os.pathsep)[0], str(tools))

    def test_incomplete_local_pair_does_not_mix_with_system(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'tools').mkdir()
            (root/'tools'/('ffmpeg.exe' if os.name=='nt' else 'ffmpeg')).touch()
            with patch.object(runtime, 'ROOT', root), patch.dict(os.environ, {'PATH':'system-tools'}):
                self.assertIsNone(runtime.configure_tools())
                self.assertEqual(os.environ['PATH'], 'system-tools')


class ReadinessTests(unittest.TestCase):
    def state(self, ready):
        return dict(status='ready' if ready else 'unavailable', recipe='spectral-face-v3', sha256='abc',
                    message='compatible' if ready else 'FFmpeg: expected 8.1.1; found 9.0.2',
                    issues=[] if ready else ['FFmpeg mismatch'])

    def test_health_exposes_existing_but_incompatible_model(self):
        with patch.object(spectral, 'spectral_status', return_value=self.state(False)):
            response = TestClient(app.app).get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['all_models_ready'])
        self.assertEqual(data['models']['spectral']['status'], 'unavailable')
        self.assertIn('FFmpeg', data['warnings'][0])

    def test_health_confirms_compatible_model(self):
        with patch.object(spectral, 'spectral_status', return_value=self.state(True)):
            data = TestClient(app.app).get('/api/health').json()
        self.assertTrue(data['all_models_ready'])
        self.assertEqual(data['warnings'], [])

    def test_preflight_rejects_weight_checksum_mismatch(self):
        with patch('demo.check_runtime.spectral_status', return_value=self.state(True)), \
             patch('demo.check_runtime.model_sources', return_value={'spatial':dict(path='model', sha256='expected')}), \
             patch('demo.check_runtime.sha256', return_value='wrong'):
            data = check_runtime()
        self.assertEqual(data['status'], 'unavailable')
        self.assertIn('spatial weights checksum mismatch', data['issues'][0])

class PackageRuntimeTests(unittest.TestCase):
    def test_incompatible_decoder_cannot_be_exported(self):
        from demo import package
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'release.zip'
            with patch.object(package, 'export_weights', return_value=Path(temporary)), \
                 patch.object(package, 'spectral_status', return_value=dict(status='unavailable', message='FFmpeg mismatch')), \
                 patch.object(package.zipfile, 'ZipFile') as archive, \
                 patch('sys.argv', ['package', '--output', str(target)]):
                with self.assertRaisesRegex(ValueError, 'Cannot export an incompatible runtime'):
                    package.main()
                archive.assert_not_called()
                self.assertFalse(target.exists())

if __name__ == '__main__':
    unittest.main()