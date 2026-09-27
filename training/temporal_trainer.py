import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.temporal_checkpoint import (
    load_temporal_checkpoint,
    save_temporal_checkpoint,
)
from training.temporal_config import (
    TEMPORAL_BEST_MODEL,
    TEMPORAL_CHECKPOINT,
    TEMPORAL_DEVICE,
    TEMPORAL_EPOCHS,
    TEMPORAL_LAST_MODEL,
    TEMPORAL_LEARNING_RATE,
    TEMPORAL_LOG_EVERY_N_BATCHES,
    TEMPORAL_USE_AMP,
    TEMPORAL_WEIGHT_DECAY,
)
from training.temporal_logger import TEMPORAL_LOGGER
from training.temporal_validator import validate_temporal


def _gpu_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "gpu_mem=n/a"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    return (
        f"gpu_alloc={allocated:.2f}GB "
        f"gpu_reserved={reserved:.2f}GB "
        f"gpu_peak={peak:.2f}GB"
    )


class TemporalTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        train_dataset=None,
        resume: bool = True,
    ):
        self.model = model.to(TEMPORAL_DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset

        self.use_amp = (
            TEMPORAL_USE_AMP
            and torch.cuda.is_available()
            and str(TEMPORAL_DEVICE).startswith("cuda")
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TEMPORAL_LEARNING_RATE,
            weight_decay=TEMPORAL_WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=2,
            factor=0.5,
        )

        if resume:
            self.start_epoch, self.best_accuracy = load_temporal_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
            )
            TEMPORAL_LOGGER.info(
                "Continue temporal training from epoch %s",
                self.start_epoch,
            )
        else:
            self.start_epoch, self.best_accuracy = 0, 0.0
            TEMPORAL_LOGGER.info("Start fresh temporal training run")

        TEMPORAL_LOGGER.info("Checkpoint path: %s", TEMPORAL_CHECKPOINT)
        TEMPORAL_LOGGER.info("Best model path : %s", TEMPORAL_BEST_MODEL)
        TEMPORAL_LOGGER.info("Last model path : %s", TEMPORAL_LAST_MODEL)
        TEMPORAL_LOGGER.info("Device          : %s", TEMPORAL_DEVICE)
        TEMPORAL_LOGGER.info("AMP enabled     : %s", self.use_amp)

    def train(self) -> None:
        try:
            for epoch in range(self.start_epoch, TEMPORAL_EPOCHS):
                epoch_start = time.time()

                TEMPORAL_LOGGER.info("=" * 72)
                TEMPORAL_LOGGER.info(
                    "Temporal epoch %s/%s",
                    epoch + 1,
                    TEMPORAL_EPOCHS,
                )

                if self.train_dataset is not None:
                    self.train_dataset.set_epoch(epoch)

                self.model.train()

                running_loss = 0.0
                correct = 0
                total = 0

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                loop = tqdm(
                    self.train_loader,
                    desc=f"TEMPORAL TRAIN {epoch + 1}",
                )

                for batch_index, batch in enumerate(loop, start=1):
                    clips, labels, _video_ids = batch

                    clips = clips.to(TEMPORAL_DEVICE)
                    labels = labels.to(TEMPORAL_DEVICE).unsqueeze(1)

                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(
                        device_type="cuda",
                        enabled=self.use_amp,
                    ):
                        outputs = self.model(clips)
                        loss = self.criterion(outputs, labels)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    running_loss += loss.item()

                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    train_acc = correct / max(total, 1)
                    avg_loss = running_loss / batch_index
                    lr = self.optimizer.param_groups[0]["lr"]

                    loop.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg=f"{avg_loss:.4f}",
                        clip_acc=f"{train_acc:.4f}",
                        lr=f"{lr:.1e}",
                    )

                    if (
                        batch_index == 1
                        or batch_index % TEMPORAL_LOG_EVERY_N_BATCHES == 0
                        or batch_index == len(self.train_loader)
                    ):
                        TEMPORAL_LOGGER.info(
                            "Train batch %s/%s | loss=%.4f avg_loss=%.4f "
                            "clip_acc=%.4f lr=%.2e %s",
                            batch_index,
                            len(self.train_loader),
                            loss.item(),
                            avg_loss,
                            train_acc,
                            lr,
                            _gpu_memory_summary(),
                        )

                train_acc = correct / max(total, 1)
                train_loss = running_loss / max(len(self.train_loader), 1)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                TEMPORAL_LOGGER.info("Validation started.")
                val = validate_temporal(
                    self.model,
                    self.val_loader,
                    self.criterion,
                    TEMPORAL_DEVICE,
                )

                elapsed = time.time() - epoch_start

                TEMPORAL_LOGGER.info(
                    "Epoch result | train_loss=%.4f train_clip_acc=%.4f "
                    "val_loss=%.4f val_video_acc=%.4f val_clip_acc=%.4f "
                    "P=%.4f R=%.4f F1=%.4f videos=%s clips=%s elapsed=%.1fs %s",
                    train_loss,
                    train_acc,
                    val["loss"],
                    val["video_accuracy"],
                    val["clip_accuracy"],
                    val["precision"],
                    val["recall"],
                    val["f1"],
                    val["videos"],
                    val["clips"],
                    elapsed,
                    _gpu_memory_summary(),
                )

                print()
                print("-" * 72)
                print(f"Train clip accuracy : {train_acc:.4f}")
                print(f"Train loss          : {train_loss:.4f}")
                print(f"Video val accuracy  : {val['video_accuracy']:.4f}")
                print(f"Clip val accuracy   : {val['clip_accuracy']:.4f}")
                print(f"Precision           : {val['precision']:.4f}")
                print(f"Recall              : {val['recall']:.4f}")
                print(f"F1-score            : {val['f1']:.4f}")
                print("-" * 72)

                torch.save(self.model.state_dict(), TEMPORAL_LAST_MODEL)

                if val["video_accuracy"] > self.best_accuracy:
                    self.best_accuracy = val["video_accuracy"]
                    torch.save(self.model.state_dict(), TEMPORAL_BEST_MODEL)
                    TEMPORAL_LOGGER.info("Best temporal model updated.")

                save_temporal_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    self.best_accuracy,
                )

                self.scheduler.step(val["video_accuracy"])

        except KeyboardInterrupt:
            TEMPORAL_LOGGER.info("Temporal training interrupted.")
            save_temporal_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_accuracy,
            )

            print()
            print("Temporal checkpoint saved.")
            print("Continue next time from the same epoch.")
