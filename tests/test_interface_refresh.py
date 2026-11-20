import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from demo import app, runtime


class RefreshTests(unittest.TestCase):
    def test_revision_is_stable_and_tracks_ui_edits(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime,'ROOT',Path(folder)):
            root=Path(folder)
            (root/'demo/web').mkdir(parents=True)
            asset=root/'demo/web/app.js'
            asset.write_text('version one')
            before=runtime.application_revision()
            self.assertEqual(runtime.application_revision(),before)
            asset.write_text('version two')
            self.assertNotEqual(runtime.application_revision(),before)

    def test_reports_do_not_change_code_revision(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime,'ROOT',Path(folder)):
            root=Path(folder)
            before=runtime.application_revision()
            (root/'demo_runtime').mkdir()
            (root/'demo_runtime/report.json').write_text('new result')
            self.assertEqual(runtime.application_revision(),before)

    def test_health_reports_boot_revision_not_new_disk_revision(self):
        with patch.object(app,'application_revision',return_value='modified-after-boot'):
            data=TestClient(app.app).get('/api/health').json()
        self.assertEqual(data['application_revision'],app.APPLICATION_REVISION)
        self.assertEqual(data['project_root'],str(app.ROOT))
        self.assertIn('busy',data)

    def test_html_and_scripts_are_not_cached(self):
        client=TestClient(app.app)
        for url in ('/','/assets/app.js?v=7','/assets/scores.js?v=1','/assets/style.css?v=7'):
            response=client.get(url)
            self.assertEqual(response.status_code,200)
            self.assertIn('no-store',response.headers['cache-control'])


if __name__=='__main__':
    unittest.main()