# AI Video Detector

## Начать здесь

Текущее состояние проекта и перенос на другой компьютер:
[HANDOFF.md](HANDOFF.md). Подробная история решений, обучения, ошибок,
результатов и оставшихся задач: [WORK_LOG.md](WORK_LOG.md).

Локальная демонстрация TRACE: `start-demo.cmd`. Спектральная ветка и
подробный отчёт: [TRACE_ENSEMBLE.md](TRACE_ENSEMBLE.md).
Установка и переносной архив описаны в [DEMO_GUIDE.md](DEMO_GUIDE.md).
Проверка двух основных моделей на независимых видео, после пережатия и по
времени вмешательства: [TRACE_BENCHMARK.md](TRACE_BENCHMARK.md).
Техническая справка для практической главы ВКР: [TRACE_THESIS_TECHNICAL.md](TRACE_THESIS_TECHNICAL.md).
Датасеты, веса, checkpoints и архив демонстрации исключены из Git:
одного клона репозитория недостаточно для запуска обученных моделей.

## Третья модель: спектр и шумовой остаток

`spectral-face-v3` берёт три кадра из разных моментов видео, переводит их в
канал яркости и извлекает 30 признаков: энергию десяти полос 2D FFT и
статистики шумового остатка, обобщённые по кадрам. OpenCV YuNet находит
лицо, если оно есть; для этой области считаются те же признаки. Отдельные
логистические классификаторы оценивают кадр и лицо, а их баллы усредняются.
Если лицо не найдено, остаётся оценка всего кадра. Модель не идентифицирует
камеру и не определяет точную область подмены.

В TRACE общий балл складывается из **63% изображения, 27% движения и 10%
спектра**, когда все три ветки доступны. Порог `0.65` для сигнала кадра и
движения задан на сервере: уведомление о возможном вмешательстве требует
совпадения этих сигналов на одном участке. Спектр уточняет балл и может
направить результат на ручную проверку, но не поднимает и не отменяет тревогу
самостоятельно. Если спектр недоступен, а первые две ветки отработали,
используется `70/30`. Это экспериментальное правило, не доказанный прирост
качества; балл не является вероятностью генерации.

Обучение, DFDC и локальный монитор: [SPECTRAL_TRAINING.md](SPECTRAL_TRAINING.md).
Подключение весов и правила отчёта: [TRACE_ENSEMBLE.md](TRACE_ENSEMBLE.md).

## Инструменты

| Назначение | Что используется |
| --- | --- |
| Сервер и видео | Python 3.12, FastAPI, Uvicorn, FFmpeg и FFprobe. |
| Модели и признаки | PyTorch и torchvision для EfficientNet V2-S и R(2+1)D-18; OpenCV для оптического потока и YuNet, NumPy для спектральных признаков. |
| Веб-интерфейс | HTML, CSS и JavaScript; Lenis для прокрутки, Lucide для иконок, шрифт Manrope. Эти ресурсы находятся в `demo/web` и не требуют CDN. |

При разработке интерфейса использовались [Impeccable](https://github.com/pbakaus/impeccable)
для дизайн-аудита, [Taste Skill](https://github.com/Leonxlnx/taste-skill) для
компоновки и адаптивности и [Emil Kowalski Skills](https://github.com/emilkowalski/skills)
(`animate`, `emil-design-eng`) для анимаций. Это инструкции для Codex, а не
зависимости сайта. Правила оформления хранятся в [DESIGN.md](DESIGN.md),
история применения инструментов — в [WORK_LOG.md](WORK_LOG.md).

## Новый spatial-классификатор

Эксперимент `hybrid-v2`: общий кадр + локальные области, сопоставление real/fake
по разрешению и пропорциям, ограничение UCF и подробная диагностика.
Запуск: `powershell -NoProfile -ExecutionPolicy Bypass -File .\train-classifier.ps1`.
Порядок продолжения и проверки описан в [HYBRID_TRAINING.md](HYBRID_TRAINING.md).
Старые веса и запуск `patch_v1` сохраняются отдельно; улучшение качества нужно измерить.

Experimental local-patch spatial fine-tuning: see [ARTIFACT_TRAINING.md](ARTIFACT_TRAINING.md).
It uses a separate run and command (`train-artifacts.ps1`); existing spatial_v2 runs remain compatible.

## Spatial training v2

The current spatial entry point uses audited, isolated runs. Start or resume:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train-spatial.ps1
```

See [SPATIAL_TRAINING.md](SPATIAL_TRAINING.md) for the current Russian guide,
dataset checks, logs, limits, checkpoints, and evaluation. The legacy spatial
instructions below describe the earlier pipeline; they are not required for v2.

Дипломный проект по разработке интеллектуальной системы обнаружения сгенерированных видеозаписей на основе ансамбля нейронных сетей.

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

## Temporal V2: обучение второй модели

Новая независимая инфраструктура R(2+1)D-18 с локальной панелью, визуализацией
клипов и Grad-CAM, подробными логами и возобновлением:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train-temporal.ps1
```

Панель: `http://127.0.0.1:8766` (точный адрес печатается при запуске).
Полное руководство: [TEMPORAL_TRAINING.md](TEMPORAL_TRAINING.md).
Старая команда `training.train_temporal` не использует новую инфраструктуру.

### Motion-режим: только признаки движения

Отдельный запуск с оптическим потоком вместо RGB-входа:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train-temporal.ps1 -MotionTrain
```

Нынешние RGB-веса сохраняются. Используются текущие видео, но отбор ограничен
длиной, исходным FPS и балансом real/fake. Команда повторно продолжает именно
motion-запуск. Панель показывает карты движения и контрольные проверки.
Подробности и ограничения: [TEMPORAL_FLOW_TRAINING.md](TEMPORAL_FLOW_TRAINING.md).

## Расположение датасетов

Архивы из `Downloads\dataset` перенесены в `D:\dataset` и доступны через
`datasets\archive`. Рабочая папка `data` и сохраненные выборки обучения не менялись.
Подробности: [DATASET_LOCATIONS.md](DATASET_LOCATIONS.md).

## TRACE: локальная демонстрация

Интерфейс с тремя аналитическими ветками, дорожкой кадров, Grad-CAM,
оптическим потоком и спектром. Подробный отчёт можно скачать как автономный
HTML с графиками и кадрами; также доступны JSON и CSV:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-demo.ps1
```

Перенос на другой компьютер и ограничения: [DEMO_GUIDE.md](DEMO_GUIDE.md).

## Старый Backend

```bash
python -m uvicorn backend.main:app --reload
```
