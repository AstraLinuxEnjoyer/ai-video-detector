# AI Video Detector

## Spatial training v2

The current spatial entry point uses audited, isolated runs. Start or resume:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train-spatial.ps1
```

See [SPATIAL_TRAINING.md](SPATIAL_TRAINING.md) for the current Russian guide,
dataset checks, logs, limits, checkpoints, and evaluation. The legacy spatial
instructions below describe the earlier pipeline; they are not required for v2.

Дипломный проект по разработке интеллектуальной системы обнаружения сгенерированных видеозаписей на основе ансамбля нейронных сетей.

## Стек

- Python 3.12
- FastAPI
- PyTorch
- React
- PostgreSQL
- FFmpeg
- Docker

## Статус

Проект находится в разработке.

## Старый Процесс Spatial-Обучения (До v2)

Датасет подтягивается автоматически с HuggingFace ([DeepAction v1](https://huggingface.co/datasets/faridlab/deepaction_v1)):

- **реальные видео** — папка `Pexels`
- **сгенерированные** — AnimateDiff, CogVideoX5B, RunwayML, StableDiffusion, Veo, VideoPoet

Этот раздел сохранён для истории. Для текущего обучения используй команду и руководство выше: автоматически докачивается DeepAction, остальные локальные источники тоже учитываются.

```bash
cd C:\Users\Ivan\Desktop\ai-video-detector
venv\Scripts\activate

# 1. Скачать только каталог и создать train/val split (без видео)
python -m training.prepare_dataset

# 2. Обучение: видео скачиваются по мере необходимости, кадры берутся на лету
python -m training.train_spatial
```

Кэш:

- `cache/spatial/catalog.json` — список всех видео на HuggingFace
- `cache/spatial/train.json`, `val.json` — train/val split
- `cache/spatial/videos/` — локальный кэш скачанных видео

Инференс по одному изображению:

```bash
python training/inference.py test_images/cat.jpg
```

## Backend

```bash
python -m uvicorn backend.main:app --reload
```
