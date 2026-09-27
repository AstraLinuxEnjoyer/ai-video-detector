import argparse

from torch.utils.data import DataLoader

from training.data.pipeline import prepare_training_manifests
from training.temporal_config import (
    TEMPORAL_BATCH_SIZE,
    TEMPORAL_CLIP_LENGTH,
    TEMPORAL_CLIPS_PER_VIDEO,
    TEMPORAL_DEVICE,
    TEMPORAL_IMAGE_SIZE,
    TEMPORAL_NUM_WORKERS,
    TEMPORAL_PIN_MEMORY,
    TEMPORAL_PRETRAINED,
    TEMPORAL_VAL_BATCH_SIZE,
    TEMPORAL_VAL_CLIPS_PER_VIDEO,
    TEMPORAL_VAL_VIDEOS,
    TEMPORAL_VIDEOS_PER_EPOCH,
)
from training.temporal_dataset import TemporalVideoDataset
from training.temporal_logger import TEMPORAL_LOGGER
from training.temporal_model import build_temporal_model
from training.temporal_trainer import TemporalTrainer


def _make_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TEMPORAL_NUM_WORKERS,
        pin_memory=TEMPORAL_PIN_MEMORY,
        persistent_workers=TEMPORAL_NUM_WORKERS > 0,
    )


def main(
    resume: bool = True,
    rebuild: bool = False,
    pretrained: bool = TEMPORAL_PRETRAINED,
) -> None:
    print()
    print("=" * 72)
    print("Temporal model training")
    print("=" * 72)
    print()
    print("Model       : R(2+1)D-18")
    print("Device      :", TEMPORAL_DEVICE)
    print("Pretrained  :", pretrained)
    print("Clip length :", TEMPORAL_CLIP_LENGTH)
    print("Image size  :", TEMPORAL_IMAGE_SIZE)
    print()

    TEMPORAL_LOGGER.info("Temporal training command started.")
    TEMPORAL_LOGGER.info("Rebuild manifests: %s", rebuild)
    TEMPORAL_LOGGER.info("Resume checkpoint: %s", resume)
    TEMPORAL_LOGGER.info("Pretrained model : %s", pretrained)
    TEMPORAL_LOGGER.info(
        "Input setup | clip_length=%s image_size=%s train_clips/video=%s "
        "val_clips/video=%s videos/epoch=%s val_videos=%s",
        TEMPORAL_CLIP_LENGTH,
        TEMPORAL_IMAGE_SIZE,
        TEMPORAL_CLIPS_PER_VIDEO,
        TEMPORAL_VAL_CLIPS_PER_VIDEO,
        TEMPORAL_VIDEOS_PER_EPOCH,
        TEMPORAL_VAL_VIDEOS,
    )

    train_entries, val_entries = prepare_training_manifests(
        rebuild=rebuild,
    )

    TEMPORAL_LOGGER.info(
        "Manifest loaded | train_videos=%s val_videos=%s",
        len(train_entries),
        len(val_entries),
    )

    train_dataset = TemporalVideoDataset(
        train_entries,
        train=True,
        epoch=0,
        max_videos=TEMPORAL_VIDEOS_PER_EPOCH,
    )

    val_dataset = TemporalVideoDataset(
        val_entries,
        train=False,
        epoch=0,
        max_videos=TEMPORAL_VAL_VIDEOS,
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size=TEMPORAL_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=TEMPORAL_VAL_BATCH_SIZE,
        shuffle=False,
    )

    TEMPORAL_LOGGER.info(
        "Loaders ready | train_batches=%s val_batches=%s batch_size=%s "
        "val_batch_size=%s workers=%s",
        len(train_loader),
        len(val_loader),
        TEMPORAL_BATCH_SIZE,
        TEMPORAL_VAL_BATCH_SIZE,
        TEMPORAL_NUM_WORKERS,
    )

    model = build_temporal_model(pretrained=pretrained)

    trainer = TemporalTrainer(
        model,
        train_loader,
        val_loader,
        train_dataset=train_dataset,
        resume=resume,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new temporal run without loading the previous checkpoint.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the shared video catalog and train/val manifests first.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize R(2+1)D-18 from scratch instead of Kinetics weights.",
    )
    args = parser.parse_args()

    main(
        resume=not args.fresh,
        rebuild=args.rebuild,
        pretrained=not args.no_pretrained,
    )
