"""Export an allowlisted, dataset-free portable demonstration ZIP."""
import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from training.spatial.common import atomic_json
from .inference import ROOT, model_sources, sha256


def export_weights():
    # Freeze the currently selected best weights, never optimizer/checkpoint state.
    info = model_sources(prefer_training=True)
    output = ROOT/'demo_weights'
    output.mkdir(exist_ok=True)
    manifest = {}
    for kind in ('spatial', 'temporal'):
        entry = info[kind]
        target = output/f'{kind}.pt'
        if entry['path'].resolve() != target.resolve():
            temporary = target.with_suffix('.pt.tmp')
            shutil.copy2(entry['path'], temporary)
            if sha256(temporary) != entry['sha256']:
                temporary.unlink()
                raise RuntimeError('Training weights changed during export; export again after saving finishes')
            os.replace(temporary, target)
        manifest[kind] = dict(file=target.name, config=entry['config'], epoch=entry['epoch'], sha256=sha256(target))
    atomic_json(output/'manifest.json', manifest)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-only', action='store_true')
    args = parser.parse_args()
    weights = export_weights()
    if args.weights_only:
        print(f'Frozen demo weights: {weights}')
        return
    release = ROOT/'releases'
    release.mkdir(exist_ok=True)
    destination = release/'TRACE-demo-windows.zip'
    # Explicit allowlist: never traverse data/datasets/cache/reports/venv or junctions.
    paths = []
    for directory in ('demo', 'training', 'demo_weights'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')
                     and (directory != 'training' or (p.suffix == '.py' and 'datasets' not in p.parts)))
    for name in ('start-demo.ps1', 'start-demo.cmd', 'setup-demo.ps1', 'setup-demo.cmd',
                 'demo-requirements.txt', 'DEMO_GUIDE.md', 'HANDOFF.md', 'WORK_LOG.md'):
        paths.append(ROOT/name)
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError('FFmpeg and FFprobe required to build the portable bundle')
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in paths:
            archive.write(path, 'TRACE-demo/'+str(path.relative_to(ROOT)).replace('\\', '/'))
        for path in (ffmpeg, ffprobe):
            archive.write(path, 'TRACE-demo/tools/'+Path(path).name)
        license_path = Path(ffmpeg).parent.parent/'LICENSE'
        if license_path.is_file():
            archive.write(license_path, 'TRACE-demo/tools/FFMPEG_LICENSE')
        license_text = subprocess.run([ffmpeg, '-L'], capture_output=True, check=True, text=True).stdout
        archive.writestr('TRACE-demo/tools/FFMPEG_BUILD_LICENSE.txt', license_text)
        archive.writestr('TRACE-demo/tools/FFMPEG_SOURCE.txt',
                        'FFmpeg bundled from https://www.gyan.dev/ffmpeg/builds/\n'
                        'FFmpeg source: https://ffmpeg.org/download.html\n'
                        'Build scripts/source information: https://github.com/GyanD/codexffmpeg\n'
                        'See FFMPEG_BUILD_LICENSE.txt for this binary configuration and license.\n')
    checksum = sha256(destination)
    atomic_json(release/'TRACE-demo-windows.sha256.json', dict(file=destination.name, sha256=checksum))
    print(f'Portable demo: {destination} ({destination.stat().st_size/1024**2:.1f} MB)')
    print('Datasets, uploads, reports, environments and training checkpoints are excluded.')


if __name__ == '__main__':
    main()
