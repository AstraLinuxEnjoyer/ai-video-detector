"""Read-only localhost dashboard. Serves only published JSON and preview images."""
import argparse
import json
import logging
import mimetypes
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from filelock import FileLock

from training.spatial.common import ROOT, atomic_json, event

try:
    import psutil
except ImportError:
    psutil = None


def read_shared(path):
    """Coordinate live snapshot readers with writers, including separate servers."""
    path = Path(path)
    if path.name == 'live.json':
        with FileLock(str(path)+'.access.lock', timeout=.5):
            return path.read_bytes()
    return path.read_bytes()


def windows_system(previous=None):
    """Read-only Windows counters; no extra package required on the training host."""
    import ctypes
    from ctypes import wintypes
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return {}, previous
    value = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime
    current = (value(idle), value(kernel)+value(user))
    cpu = None
    if previous and current[1] > previous[1]:
        cpu = max(0., min(100., 100*(1-(current[0]-previous[0])/(current[1]-previous[1]))))
    class Memory(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)]+[
            (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available',
                                                   'virtual_total', 'virtual_available', 'extended')]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    return dict(cpu_percent=cpu, ram_percent=memory.load if ok else None), current


class Monitor:
    def __init__(self, run):
        self.run = Path(run)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.log = logging.getLogger('spatial_v2')
        self.failures = {}
        self.last_warning = {}
        path = self.run/'live.json'
        self.data = {}
        try:
            if path.exists():
                self.data = json.loads(read_shared(path))
        except (OSError, ValueError) as exc:
            self.log.warning('Dashboard snapshot unavailable; rebuilding telemetry: %s', exc)
        self.thread = threading.Thread(target=self._system, daemon=True)
        self.thread.start()

    def _system(self):
        previous = None
        while not self.stopped.is_set():
            system = dict(cpu_percent=psutil.cpu_percent() if psutil else None,
                          ram_percent=psutil.virtual_memory().percent if psutil else None)
            if psutil is None and os.name == 'nt':
                counters, previous = windows_system(previous)
                system.update(counters)
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                values = [float(v.strip()) for v in result.stdout.splitlines()[0].split(',')]
                system.update(gpu_percent=values[0], gpu_used_mb=values[1], gpu_total_mb=values[2], gpu_c=values[3])
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                system['gpu_percent'] = None
            self.publish(system=system)
            self.stopped.wait(5)

    def publish(self, **values):
        with self.lock:
            self.data.update(values, updated=time.time(), run=str(self.run))
            return self._write('live.json', self._snapshot)

    def _snapshot(self):
        path = self.run/'live.json'
        with FileLock(str(path)+'.access.lock', timeout=.25):
            atomic_json(path, self.data)

    def _write(self, name, write):
        # Telemetry is optional; never apply this policy to weights or metrics.
        try:
            write()
        except OSError as exc:
            self.failures[name] = self.failures.get(name, 0) + 1
            now = time.monotonic()
            if name not in self.last_warning or now-self.last_warning[name] >= 30:
                self.log.warning('TELEMETRY %s unavailable (%d updates deferred); training continues: %s',
                                 name, self.failures[name], exc)
                self.last_warning[name] = now
            return False
        if name in self.failures:
            self.log.info('TELEMETRY %s recovered after %d deferred updates', name, self.failures.pop(name))
            self.last_warning.pop(name, None)
        return True

    def record(self, **values):
        self.publish(**values)
        with self.lock:
            self._write('performance.jsonl', lambda: event(self.run/'performance.jsonl',
                        dict(values, time=time.time(), system=self.data.get('system'))))

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=5)


def handler_for(run):
    root = Path(run).resolve()
    assets = Path(__file__).parent/'web'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == '/':
                path = assets/'index.html'
            elif route in ('/app.js', '/style.css'):
                path = assets/route[1:]
            elif route == '/api/state':
                path = root/'live.json'
            elif route == '/api/history':
                path = root/'history.json'
            elif re.fullmatch(r'/api/explanations/\d{1,4}', route):
                path = root/f"explanations_{int(route.rsplit('/', 1)[1]):03}.json"
            elif route.startswith('/previews/'):
                path = (root/route.lstrip('/')).resolve()
                if not path.is_relative_to((root/'previews').resolve()) or path.suffix not in ('.jpg', '.gif', '.png'):
                    self.send_error(403)
                    return
            else:
                self.send_error(404)
                return
            try:
                content = read_shared(path)
            except FileNotFoundError:
                if route.startswith('/api/'):
                    content = b'{}' if route.endswith('state') else b'[]'
                else:
                    self.send_error(404)
                    return
            except OSError:
                self.send_error(503, 'Snapshot temporarily unavailable')
                return
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def start_server(run, port=8766):
    for candidate in range(port, port+20):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', candidate), handler_for(run))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return server, f'http://127.0.0.1:{server.server_port}'
        except OSError:
            continue
    raise OSError('No free local dashboard port')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT/'cache/temporal_v2/runs/r2plus1d_v1')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server, url = start_server(args.run_dir, args.port)
    print('Temporal dashboard:', url, flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
