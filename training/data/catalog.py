import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

from training.config import (
    CATALOG_PATH,
    DATASET_REPO_ID,
    FAKE_GENERATORS,
    LOCAL_FAKE_DIR,
    LOCAL_REAL_DIR,
    MIN_VIDEO_BYTES,
    REMOTE_DATASETS,
    ROOT,
    REAL_SOURCE,
    VIDEO_EXTENSIONS,
)


@dataclass(frozen=True)
class VideoEntry:
    id: str
    repo_path: str
    label: int
    source: str
    group_id: str = ""
    local_path: str | None = None
    repo_id: str | None = None
    repo_type: str = "dataset"
    dataset: str = ""

    @property
    def is_real(self) -> bool:
        return self.label == 0

    @property
    def is_local(self) -> bool:
        return self.local_path is not None

    def __post_init__(self) -> None:
        if not self.group_id:
            object.__setattr__(
                self,
                "group_id",
                _make_group_id(self.repo_path, self.source),
            )
        if self.local_path == "":
            object.__setattr__(self, "local_path", None)
        if self.repo_id == "":
            object.__setattr__(self, "repo_id", None)
        if self.repo_id is None and self.local_path is None:
            object.__setattr__(self, "repo_id", DATASET_REPO_ID)
        if not self.dataset:
            dataset = "local" if self.local_path is not None else "deepaction_v1"
            object.__setattr__(self, "dataset", dataset)


def _make_entry_id(repo_path: str, namespace: str = "") -> str:
    raw = f"{namespace}:{repo_path}" if namespace else repo_path
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{digest}_{Path(repo_path).stem}"


def _make_group_id(repo_path: str, source: str) -> str:
    parts = Path(repo_path).parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return source


def _parse_repo_path(repo_path: str, dataset_spec: dict) -> VideoEntry | None:
    parts = Path(repo_path).parts
    if len(parts) < 3:
        return None

    source = parts[0]
    suffix = Path(repo_path).suffix.lower()

    if suffix not in VIDEO_EXTENSIONS:
        return None

    real_sources = set(dataset_spec.get("real_sources", [REAL_SOURCE]))
    fake_sources = set(dataset_spec.get("fake_sources", FAKE_GENERATORS))

    if source in real_sources:
        label = 0
    elif source in fake_sources:
        label = 1
    else:
        return None

    repo_id = dataset_spec["repo_id"]
    dataset_name = dataset_spec["name"]

    return VideoEntry(
        id=_make_entry_id(repo_path, namespace=repo_id),
        repo_path=repo_path.replace("\\", "/"),
        label=label,
        source=f"{dataset_name}:{source}",
        group_id=f"{dataset_name}/" + _make_group_id(repo_path, source),
        repo_id=repo_id,
        repo_type=dataset_spec.get("repo_type", "dataset"),
        dataset=dataset_name,
    )


def _limit_entries(entries: list[VideoEntry], dataset_spec: dict) -> list[VideoEntry]:
    max_real = dataset_spec.get("max_real_videos")
    max_fake = dataset_spec.get("max_fake_videos")

    if max_real is None and max_fake is None:
        return entries

    real = [entry for entry in entries if entry.is_real]
    fake = [entry for entry in entries if not entry.is_real]

    if max_real is not None:
        real = real[: int(max_real)]
    if max_fake is not None:
        fake = fake[: int(max_fake)]

    return [*real, *fake]


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _local_group_id(class_name: str, relative_path: Path) -> str:
    parts = relative_path.parts
    if len(parts) >= 2:
        return f"local/{class_name}/" + "/".join(parts[:2])
    return f"local/{class_name}/{relative_path.stem}"


def _scan_local_dir(root: Path, label: int, class_name: str) -> list[VideoEntry]:
    if not root.exists():
        return []

    entries: list[VideoEntry] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if path.stat().st_size < MIN_VIDEO_BYTES:
            continue

        relative = path.relative_to(root)
        source_name = relative.parts[0] if len(relative.parts) > 1 else "local"
        local_path = _relative_to_root(path)
        repo_path = f"local/{class_name}/{relative}".replace("\\", "/")

        entries.append(
            VideoEntry(
                id=_make_entry_id(local_path),
                repo_path=repo_path,
                label=label,
                source=f"local:{source_name}",
                group_id=_local_group_id(class_name, relative),
                local_path=local_path,
                repo_id=None,
                dataset="local",
            )
        )

    return entries


def scan_local_videos() -> list[VideoEntry]:
    return [
        *_scan_local_dir(LOCAL_REAL_DIR, label=0, class_name="real"),
        *_scan_local_dir(LOCAL_FAKE_DIR, label=1, class_name="fake"),
    ]


def fetch_catalog_from_hf(dataset_spec: dict | None = None) -> list[VideoEntry]:
    if HfApi is None:
        raise RuntimeError(
            "huggingface_hub is required to refresh remote dataset catalogs."
        )

    if dataset_spec is None:
        dataset_spec = {
            "name": "deepaction_v1",
            "repo_id": DATASET_REPO_ID,
            "repo_type": "dataset",
            "real_sources": [REAL_SOURCE],
            "fake_sources": FAKE_GENERATORS,
        }

    api = HfApi()
    files = api.list_repo_files(
        dataset_spec["repo_id"],
        repo_type=dataset_spec.get("repo_type", "dataset"),
    )

    entries: list[VideoEntry] = []

    for repo_path in files:
        entry = _parse_repo_path(repo_path, dataset_spec)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda item: (item.label, item.source, item.repo_path))
    entries = _limit_entries(entries, dataset_spec)

    if not entries:
        raise RuntimeError(
            f"No videos found in HuggingFace dataset {dataset_spec['repo_id']}"
        )

    return entries


def fetch_remote_catalogs() -> list[VideoEntry]:
    entries: list[VideoEntry] = []

    for dataset_spec in REMOTE_DATASETS:
        if not dataset_spec.get("enabled", True):
            continue

        print(f"Fetching video catalog from {dataset_spec['repo_id']}...")
        entries.extend(fetch_catalog_from_hf(dataset_spec))

    return entries


def save_catalog(entries: list[VideoEntry], path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(entry) for entry in entries]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_catalog(path: Path = CATALOG_PATH) -> list[VideoEntry]:
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found at {path}. Run `python training/prepare_dataset.py` first."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    return [VideoEntry(**item) for item in raw]


def ensure_catalog(refresh: bool = False) -> list[VideoEntry]:
    if CATALOG_PATH.exists() and not refresh:
        cached = load_catalog()
    else:
        cached = fetch_remote_catalogs()

    local_entries = scan_local_videos()

    merged: dict[str, VideoEntry] = {}
    for entry in cached:
        if not entry.is_local:
            merged[entry.id] = entry
    for entry in local_entries:
        merged[entry.id] = entry

    entries = sorted(
        merged.values(),
        key=lambda item: (item.label, item.source, item.repo_path),
    )
    save_catalog(entries)

    real_count = sum(1 for entry in entries if entry.is_real)
    fake_count = len(entries) - real_count
    local_count = sum(1 for entry in entries if entry.is_local)
    print(
        "Catalog saved: "
        f"{len(entries)} videos ({real_count} real, {fake_count} fake, "
        f"{local_count} local)"
    )

    return entries
