from pathlib import Path

# ===========================
# Пути
# ===========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

REAL_DIR = DATA_DIR / "real"

FAKE_DIR = DATA_DIR / "fake"

PREPARED_DIR = DATA_DIR / "prepared"

LOG_DIR = PROJECT_ROOT / "training" / "logs"

CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"


# ===========================
# Подготовка изображений
# ===========================

IMAGE_SIZE = 384

FPS = 2

MAX_FRAMES_PER_VIDEO = 12

JPEG_QUALITY = 95


# ===========================
# Dataset
# ===========================

TRAIN_SPLIT = 0.8

VAL_SPLIT = 0.2

RANDOM_SEED = 42


# ===========================
# Форматы
# ===========================

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
)