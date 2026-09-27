from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from training.config import (
    PIN_MEMORY,
    RANDOM_SEED,
    ROOT,
)


TEMPORAL_CACHE_DIR = ROOT / "cache" / "temporal"
TEMPORAL_LOG_DIR = TEMPORAL_CACHE_DIR / "logs"
TEMPORAL_CHECKPOINT_DIR = TEMPORAL_CACHE_DIR / "checkpoints"
TEMPORAL_MODEL_DIR = ROOT / "models"

TEMPORAL_BEST_MODEL = TEMPORAL_MODEL_DIR / "temporal_best.pt"
TEMPORAL_LAST_MODEL = TEMPORAL_MODEL_DIR / "temporal_last.pt"
TEMPORAL_CHECKPOINT = TEMPORAL_CHECKPOINT_DIR / "checkpoint.pt"

if torch is not None:
    TEMPORAL_DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    TEMPORAL_DEVICE = "cpu"

TEMPORAL_MODEL_NAME = "r2plus1d_18"
TEMPORAL_PRETRAINED = True

# Video models are much heavier than image models. These defaults are chosen
# for an 8 GB GPU and can be raised later if there is headroom.
TEMPORAL_IMAGE_SIZE = 112
TEMPORAL_CLIP_LENGTH = 16
TEMPORAL_FRAME_STRIDE = 2

TEMPORAL_EPOCHS = 12
TEMPORAL_BATCH_SIZE = 4
TEMPORAL_VAL_BATCH_SIZE = 4
TEMPORAL_LEARNING_RATE = 1e-4
TEMPORAL_WEIGHT_DECAY = 1e-4
TEMPORAL_USE_AMP = True

TEMPORAL_VIDEOS_PER_EPOCH = 500
TEMPORAL_CLIPS_PER_VIDEO = 2
TEMPORAL_VAL_VIDEOS = 1000
TEMPORAL_VAL_CLIPS_PER_VIDEO = 3
TEMPORAL_VIDEO_TOP_K = 2

TEMPORAL_NUM_WORKERS = 2
TEMPORAL_PIN_MEMORY = PIN_MEMORY
TEMPORAL_LOG_EVERY_N_BATCHES = 10

TEMPORAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
