"""Exercise real localhost inference, short-video abstention and cancellation."""
import json
import time
from pathlib import Path

import httpx

BASE = 'http://127.0.0.1:8787'
ROOT = Path(__file__).resolve().parents[1]


def wait(client, job_id):
    for _ in range(240):
        value = client.get(f'/api/jobs/{job_id}').json()
        if value['status'] not in ('running', 'queued'):
            return value
        time.sleep(.25)
    raise TimeoutError('Job did not terminate')


def submit(client, path):
    with path.open('rb') as stream:
        response = client.post('/api/jobs', files={'file': (path.name, stream, 'video/mp4')})
    response.raise_for_status()
    return response.json()['id']


def main():
    with httpx.Client(base_url=BASE, timeout=30) as client:
        short = wait(client, submit(client, ROOT/'demo_runtime/short.mp4'))
        assert short['status'] == 'complete', short
        assert short['summary']['temporal'] is None and short['warnings']
        cancel_id = submit(client, ROOT/'demo_runtime/smoke.mp4')
        assert client.post(f'/api/jobs/{cancel_id}/cancel').status_code == 200
        cancelled = wait(client, cancel_id)
        assert cancelled['status'] == 'cancelled', cancelled
        print(json.dumps(dict(short_status=short['status'], temporal_abstention=short['summary']['temporal'],
                              cancel_status=cancelled['status']), indent=2))


if __name__ == '__main__':
    main()
