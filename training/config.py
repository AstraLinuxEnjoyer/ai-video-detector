from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = ROOT / "cache" / "spatial"
VIDEO_CACHE_DIR = CACHE_DIR / "videos"
CATALOG_PATH = CACHE_DIR / "catalog.json"

LOCAL_REAL_DIR = ROOT / "data" / "real"
LOCAL_FAKE_DIR = ROOT / "data" / "fake"

LOG_DIR = CACHE_DIR / "logs"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
MODEL_DIR = ROOT / "models"

TRAIN_MANIFEST = CACHE_DIR / "train.json"
VAL_MANIFEST = CACHE_DIR / "val.json"


# -----------------------------
# DATASETS
# -----------------------------

DATASET_REPO_ID = "faridlab/deepaction_v1"
REAL_SOURCE = "Pexels"
FAKE_GENERATORS = [
    "BDAnimateDiffLightning",
    "CogVideoX5B",
    "RunwayML",
    "StableDiffusion",
    "Veo",
    "VideoPoet",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_VIDEO_BYTES = 1024

REMOTE_DATASETS = [
    {
        "name": "deepaction_v1",
        "repo_id": "faridlab/deepaction_v1",
        "repo_type": "dataset",
        "enabled": True,
        "real_sources": ["Pexels"],
        "fake_sources": [
            "BDAnimateDiffLightning",
            "CogVideoX5B",
            "RunwayML",
            "StableDiffusion",
            "Veo",
            "VideoPoet",
        ],
        # Keeps experiments bounded while preserving all six fake generators.
        "max_real_videos": 1000,
        "max_fake_videos": 3000,
    },
]

# Dataset candidates that are useful, but require a separate account, form,
# token, or non-HuggingFace downloader. They are not auto-downloaded here.
MANUAL_DATASET_CANDIDATES = [
    "GenVideo / GenVideo-100K",
    "GenVidBench",
    "MVAD",
    "GenWorld",
    "DFDC",
    "FaceForensics++",
    "Celeb-DF",
]

# Keeps a single large source from dominating the train/val pool.
# Source names are produced as:
# - local:<first-folder-under-data-real-or-fake>
# - <remote_dataset_name>:<source-folder>
SOURCE_MAX_VIDEOS = {
    "local:UCF-101": 1000,
    "local:VideoUFO": 3000,
}

# Prefetch limits. Training can still lazily download cached remote videos.
PREFETCH_MAX_VIDEOS = 1000
PREFETCH_MAX_BYTES = 20 * 1024**3
MIN_FREE_DISK_BYTES = 10 * 1024**3

RANDOM_SEED = 42
TRAIN_RATIO = 0.8


# -----------------------------
# TRAIN
# -----------------------------

if torch is not None:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    DEVICE = "cpu"

IMAGE_SIZE = 384
PRESERVE_ASPECT_RATIO = True
FRAME_PAD_COLOR = 0
BATCH_SIZE = 16
VAL_BATCH_SIZE = 8
USE_AMP = True
LEARNING_RATE = 3e-4
EPOCHS = 20

# how many videos to sample per training epoch
VIDEOS_PER_EPOCH = 500

# random frames extracted per video on each epoch
FRAMES_PER_VIDEO = 5

# fixed frames used per validation video
VAL_FRAMES_PER_VIDEO = 8

# aggregate video probability from the most suspicious frames
VIDEO_TOP_K = 3

NUM_WORKERS = 4
PIN_MEMORY = True


# -----------------------------
# SAVE
# -----------------------------

BEST_MODEL = MODEL_DIR / "spatial_best.pt"
LAST_MODEL = MODEL_DIR / "spatial_last.pt"
CHECKPOINT = CHECKPOINT_DIR / "checkpoint.pt"


# -----------------------------
# CREATE DIRS
# -----------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
