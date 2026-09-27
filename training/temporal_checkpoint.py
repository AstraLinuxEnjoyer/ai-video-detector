import random

import numpy as np
import torch

from training.temporal_config import TEMPORAL_CHECKPOINT


def save_temporal_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_accuracy,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.get_rng_state(),
            "cuda_random": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        TEMPORAL_CHECKPOINT,
    )


def load_temporal_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
) -> tuple[int, float]:
    if not TEMPORAL_CHECKPOINT.exists():
        return 0, 0.0

    checkpoint = torch.load(
        TEMPORAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    random.setstate(checkpoint["python_random"])
    np.random.set_state(checkpoint["numpy_random"])
    torch.set_rng_state(checkpoint["torch_random"])

    if torch.cuda.is_available() and checkpoint["cuda_random"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random"])

    return checkpoint["epoch"], checkpoint["best_accuracy"]
