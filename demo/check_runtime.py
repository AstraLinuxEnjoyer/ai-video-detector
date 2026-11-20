"""Read-only startup preflight: frozen weights and actual preprocessing runtime."""
import argparse
import json
import sys

from .inference import model_sources, sha256
from .spectral import spectral_status


def check_runtime():
    spectral = spectral_status()
    issues = list(spectral["issues"])
    models = {}
    try:
        for kind, entry in model_sources().items():
            actual = sha256(entry["path"])
            valid = actual == entry["sha256"]
            models[kind] = dict(ready=valid, sha256=actual)
            if not valid:
                issues.append(f"{kind} weights checksum mismatch; restore the frozen weights (git lfs pull).")
    except (OSError, ValueError, KeyError) as exc:
        issues.append(str(exc))
    return dict(status="ready" if not issues else "unavailable", models=models,
                spectral=spectral, issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_runtime()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["status"] == "ready":
        print("TRACE runtime ready: all three models and the frozen decoder contract are verified.")
        print("FFmpeg: " + result["spectral"]["paths"]["ffmpeg"])
    else:
        print("TRACE runtime is incompatible:", file=sys.stderr)
        for issue in result["issues"]:
            print(" - " + issue, file=sys.stderr)
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
