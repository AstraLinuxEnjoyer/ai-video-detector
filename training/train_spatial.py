import argparse

from torch.utils.data import DataLoader

from training.config import (
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    VAL_BATCH_SIZE,
)
from training.data.pipeline import build_datasets
from training.model import build_model
from training.trainer import Trainer


def main(resume: bool = True):
    print()
    print("=" * 60)
    print("Spatial model training (remote dataset)")
    print("=" * 60)
    print()
    print("Device:", DEVICE)
    print()

    train_dataset, val_dataset = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = build_model()

    trainer = Trainer(
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
        help="Start a new training run without loading the previous checkpoint.",
    )
    args = parser.parse_args()

    main(resume=not args.fresh)
