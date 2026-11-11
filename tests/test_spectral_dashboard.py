import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from training.spatial.common import atomic_json
from training.spectral.dashboard import server_for


class DashboardTests(unittest.TestCase):
    def test_local_dashboard_before_and_after_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "not_started"
            with server_for(run, 0) as server:
                worker = threading.Thread(target=server.serve_forever)
                worker.start()
                try:
                    url = f"http://127.0.0.1:{server.server_port}"
                    with urlopen(url + "/api/state", timeout=5) as response:
                        self.assertEqual(json.load(response), {})
                        self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertFalse(run.exists(), "Opening the UI must not create a run")
                    atomic_json(run / "live.json", dict(status="running", completed=1))
                    with urlopen(url + "/api/state", timeout=5) as response:
                        self.assertEqual(json.load(response)["completed"], 1)
                    with urlopen(url + "/", timeout=5) as response:
                        self.assertIn(b'id="spectrum"', response.read())
                    for path, host, status in (("/api/state", "other.example", 403),
                                               ("/model.json", f"127.0.0.1:{server.server_port}", 404)):
                        with self.assertRaises(HTTPError) as error:
                            urlopen(Request(url + path, headers={"Host": host}), timeout=5)
                        self.assertEqual(error.exception.code, status)
                finally:
                    server.shutdown()
                    worker.join()
