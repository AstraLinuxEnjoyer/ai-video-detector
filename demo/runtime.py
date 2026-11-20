"""Choose a project-local decoder pair without changing the Windows environment."""
import os
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_tools():
    tools = ROOT / "tools"
    suffix = ".exe" if os.name == "nt" else ""
    if all((tools / (name + suffix)).is_file() for name in ("ffmpeg", "ffprobe")):
        current = os.environ.get("PATH", "")
        first = current.split(os.pathsep)[0]
        if os.path.normcase(first) != os.path.normcase(str(tools)):
            os.environ["PATH"] = str(tools) + os.pathsep + current
        return tools
    return None


def application_revision():
    """Fingerprint code/resources, not changing videos, reports or caches."""
    digest=hashlib.sha256()
    paths=[ROOT/'start-demo.ps1',ROOT/'demo-requirements.txt']
    paths += list((ROOT/'demo').rglob('*.py'))
    paths += [p for p in (ROOT/'demo/web').rglob('*')
              if p.is_file() and p.suffix in {'.html','.js','.css'}]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).replace('\\','/').encode())
        digest.update(b'\0')
        digest.update(path.read_bytes() if path.is_file() else b'missing')
        digest.update(b'\0')
    return digest.hexdigest()[:16]
