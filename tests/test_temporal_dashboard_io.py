"""Regression tests for Windows reader/writer contention and optional telemetry."""
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

from training.spatial import common
from training.temporal_v2 import dashboard


class DashboardIOTests(unittest.TestCase):
    def test_atomic_json_retries_transient_access_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            common.atomic_json(path, {'version': 1})
            replace = os.replace
            calls = []

            def busy_then_replace(source, destination):
                calls.append(source)
                if len(calls) < 3:
                    raise PermissionError('sharing violation')
                replace(source, destination)

            with patch.object(common.os, 'replace', busy_then_replace), patch.object(common.time, 'sleep'):
                common.atomic_json(path, {'version': 2})
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(path.read_text()), {'version': 2})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_atomic_json_permanent_error_is_still_fatal_and_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            common.atomic_json(path, {'version': 1})
            with patch.object(common.os, 'replace', side_effect=PermissionError('busy')), patch.object(common.time, 'sleep'):
                with self.assertRaises(PermissionError):
                    common.atomic_json(path, {'version': 2})
            self.assertEqual(json.loads(path.read_text()), {'version': 1})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_telemetry_failure_is_nonfatal_throttled_and_recovers_latest_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard.Monitor, '_system', lambda self: None):
            monitor = dashboard.Monitor(directory)
            try:
                monitor.publish(epoch=1)
                with patch.object(dashboard, 'atomic_json', side_effect=PermissionError('busy')), \
                     patch.object(dashboard, 'event', side_effect=OSError('journal unavailable')), \
                     self.assertLogs('spatial_v2', level='WARNING') as logs:
                    self.assertFalse(monitor.publish(epoch=2))
                    self.assertFalse(monitor.publish(epoch=3))
                    monitor.record(done=5)
                self.assertEqual(len(logs.output), 2)
                self.assertEqual(json.loads((Path(directory)/'live.json').read_text())['epoch'], 1)
                with self.assertLogs('spatial_v2', level='INFO') as recovery:
                    monitor.record(done=6)
                self.assertEqual(len(recovery.output), 2)
                snapshot = json.loads((Path(directory)/'live.json').read_text())
                self.assertEqual((snapshot['epoch'], snapshot['done']), (3, 6))
                self.assertEqual(monitor.failures, {})
            finally:
                monitor.close()

    def test_invalid_previous_snapshot_does_not_block_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard.Monitor, '_system', lambda self: None):
            path = Path(directory)/'live.json'
            path.write_text('{broken')
            with self.assertLogs('spatial_v2', level='WARNING'):
                monitor = dashboard.Monitor(directory)
            try:
                self.assertTrue(monitor.publish(status='running'))
                self.assertEqual(json.loads(path.read_text())['status'], 'running')
            finally:
                monitor.close()

    def test_slow_http_reader_defers_telemetry_without_attempting_replace(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard.Monitor, '_system', lambda self: None):
            path = Path(directory)/'live.json'
            common.atomic_json(path, {'version': 1})
            monitor = dashboard.Monitor(directory)
            opened, release = threading.Event(), threading.Event()
            read_bytes = Path.read_bytes

            def held_read(file):
                opened.set()
                if not release.wait(5):
                    raise TimeoutError('test reader was not released')
                return read_bytes(file)

            with ThreadPoolExecutor(max_workers=1) as pool:
                with patch.object(Path, 'read_bytes', held_read):
                    future = pool.submit(dashboard.read_shared, path)
                    try:
                        self.assertTrue(opened.wait(3))
                        with patch.object(common.os, 'replace', side_effect=AssertionError('reader still active')), \
                             self.assertLogs('spatial_v2', level='WARNING'):
                            self.assertFalse(monitor.publish(version=2))
                    finally:
                        release.set()
                    self.assertEqual(json.loads(future.result()), {'version': 1})
            try:
                self.assertTrue(monitor.publish())
                self.assertEqual(json.loads(dashboard.read_shared(path))['version'], 2)
            finally:
                monitor.close()

    def test_live_http_readers_and_concurrent_monitor_updates(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard.Monitor, '_system', lambda self: None):
            monitor = dashboard.Monitor(directory)
            monitor.publish(sequence=0, payload='x'*4000)
            server, url = dashboard.start_server(directory, 0)

            def read_many():
                for _ in range(25):
                    with urlopen(url+'/api/state', timeout=5) as response:
                        value = json.load(response)
                    self.assertEqual(value['payload'], 'x'*4000)
                    self.assertIsInstance(value['sequence'], int)

            def write_many():
                for i in range(50):
                    self.assertTrue(monitor.publish(sequence=i))

            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    tasks = [pool.submit(read_many) for _ in range(2)]+[pool.submit(write_many) for _ in range(2)]
                    for task in tasks:
                        task.result()
            finally:
                monitor.close()
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
