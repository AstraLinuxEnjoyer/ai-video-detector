"""Send three controlled edits through a running local release of TRACE."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--output', type=Path, default=Path('reports/trace_release_qa/segments'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    real = Path('test_videos/real/3.mp4')
    other_real = Path('test_videos/real/2.mp4')
    fake = Path('test_videos/fake/336852_small.mp4')
    results = []
    with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', timeout=30, trust_env=False) as client:
        for name, center, length, edit in [('real_control', other_real, 4, None),
                                            ('generated_4s', fake, 4, [4, 8]),
                                            ('generated_1s', fake, 1, [4, 5])]:
            output = args.output/f'{name}.mp4'
            chain = 'fps=24,scale=640:360,setsar=1,format=yuv420p,setpts=PTS-STARTPTS'
            graph = (f'[0:v]{chain}[a];[1:v]{chain}[b];[2:v]{chain}[c];'
                     '[a][b][c]concat=n=3:v=1:a=0[out]')
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '1',
                            '-t', '4', '-i', str(real), '-t', str(length), '-i', str(center),
                            '-ss', '5', '-t', '4', '-i', str(real), '-filter_complex_threads', '1',
                            '-filter_complex', graph, '-map', '[out]', '-c:v', 'libx264', '-threads', '1',
                            '-preset', 'fast', '-crf', '20', str(output)], check=True, timeout=60)
            with output.open('rb') as stream:
                response = client.post('/api/jobs', files={'file': (output.name, stream, 'video/mp4')})
                response.raise_for_status()
            job_id = response.json()['id']
            deadline = time.monotonic()+120
            while time.monotonic() < deadline:
                response = client.get(f'/api/jobs/{job_id}')
                response.raise_for_status()
                job = response.json()
                if job['status'] in ('complete', 'error', 'cancelled'):
                    break
                time.sleep(.5)
            else:
                raise TimeoutError(f'TRACE job {job_id} timed out')
            if job['status'] != 'complete':
                raise RuntimeError(job['message'])
            summary = job['summary']
            result = dict(case=name, ground_truth=edit, id=job_id, duration=job['metadata']['duration'],
                          verdict=summary['verdict'], reasons=summary['reasons'], segments=summary['segments'],
                          review_segments=summary['review_segments'],
                          overlap_seconds=sum(max(0, min(s['end'], edit[1])-max(s['start'], edit[0]))
                                              for s in summary['segments']) if edit else None)
            results.append(result)
            print(json.dumps(result), flush=True)
    (args.output/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
