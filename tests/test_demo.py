import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from demo import app as module
from demo.analysis import segments


class DemoTests(unittest.TestCase):
    def test_spans_merge_and_clip_at_duration(self):
        frames = [dict(time=0, spatial=.9), dict(time=2, spatial=.8), dict(time=8, spatial=.2)]
        clips = [dict(start=3, end=5, temporal=.7), dict(start=8.5, end=10, temporal=.8)]
        self.assertEqual(segments(frames, clips, 10), [dict(start=0, end=5), dict(start=8.5, end=10)])
        self.assertEqual(segments(frames, [], 1), [dict(start=0, end=1)])

    def test_empty_spans(self):
        self.assertEqual(segments([], [], 10), [])

    def test_job_snapshot_does_not_alias(self):
        with tempfile.TemporaryDirectory() as root:
            job = module.Job(Path(root), 'video.mp4')
            frames = [dict(spatial=.4)]
            job.update(frames=frames)
            frames[0]['spatial'] = 1
            snapshot = job.snapshot()
            snapshot['frames'][0]['spatial'] = 0
            self.assertEqual(job.snapshot()['frames'][0]['spatial'], .4)

    def test_local_api_and_file_security(self):
        with tempfile.TemporaryDirectory() as root, patch.object(module, 'RUNTIME', Path(root)):
            client = TestClient(module.app)
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/', headers={'host':'evil.example'}).status_code, 403)
            self.assertEqual(client.post('/api/jobs/no/cancel', headers={'origin':'https://evil.example'}).status_code, 403)
            self.assertEqual(client.post('/api/jobs', files={'file':('file.txt', b'hello')}).status_code, 415)
            self.assertEqual(client.post('/api/jobs', headers={'content-length':str(module.MAX_BYTES+2000000)}).status_code, 413)
            self.assertEqual(client.get('/api/jobs/not-a-job').status_code, 404)
            job_id = 'a'*32
            directory = Path(root)/job_id
            directory.mkdir()
            (directory/'input.video').write_bytes(b'private')
            (directory/'preview.mp4').write_bytes(b'0123456789')
            self.assertEqual(client.get(f'/api/jobs/{job_id}/files/input.video').status_code, 404)
            response = client.get(f'/api/jobs/{job_id}/files/preview.mp4', headers={'range':'bytes=2-4'})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b'234')
            self.assertEqual(client.get(f'/api/jobs/{job_id}/files/status.json').status_code, 404)


if __name__ == '__main__':
    unittest.main()
