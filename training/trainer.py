import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from training.logger import LOGGER
from training.validator import validate
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from training.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    BEST_MODEL,
    LAST_MODEL,
    USE_AMP,
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        train_dataset=None,
        resume: bool = True,
    ):

        self.model = model.to(DEVICE)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.use_amp = (
            USE_AMP
            and torch.cuda.is_available()
            and str(DEVICE).startswith("cuda")
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp,
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=2,
            factor=0.5,
        )

        if resume:
            self.start_epoch, self.start_batch, self.best_accuracy = load_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
            )
        else:
            self.start_epoch, self.start_batch, self.best_accuracy = 0, 0, 0.0

        if resume:
            LOGGER.info(
                f"Continue from epoch {self.start_epoch}"
            )
        else:
            LOGGER.info("Start fresh training run")

    def train(self):

        try:

            for epoch in range(
                self.start_epoch,
                EPOCHS,
            ):

                LOGGER.info(
                    f"Epoch {epoch+1}/{EPOCHS}"
                )

                if self.train_dataset is not None:
                    self.train_dataset.set_epoch(epoch)

                self.model.train()

                running_loss = 0

                correct = 0
                total = 0

                loop = tqdm(
                    self.train_loader,
                    desc=f"TRAIN {epoch+1}",
                )

                resume_batch = (
                    self.start_batch
                    if epoch == self.start_epoch
                    else 0
                 )

                for batch in loop:

                    images, labels = batch[0], batch[1]

                    images = images.to(DEVICE)

                    labels = (
                        labels
                        .to(DEVICE)
                        .unsqueeze(1)
                    )

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast(
                        enabled=self.use_amp,
                    ):

                        outputs = self.model(images)

                        loss = self.criterion(
                            outputs,
                            labels,
                        )

                    self.scaler.scale(loss).backward()

                    self.scaler.step(self.optimizer)

                    self.scaler.update()

                    running_loss += loss.item()

                    probs = torch.sigmoid(outputs)

                    preds = probs > 0.5

                    correct += (
                        preds == labels
                    ).sum().item()

                    total += labels.size(0)

                    loop.set_postfix(

                        loss=f"{loss.item():.4f}",

                        acc=f"{correct/total:.4f}"

                    )

                train_acc = correct / total

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                val = validate(

                    self.model,

                    self.val_loader,

                    self.criterion,

                    DEVICE,

                )

                LOGGER.info(

                    f"Train={train_acc:.4f} "

                    f"VideoVal={val['video_accuracy']:.4f} "

                    f"FrameVal={val['frame_accuracy']:.4f} "

                    f"P={val['precision']:.4f} "

                    f"R={val['recall']:.4f} "

                    f"F1={val['f1']:.4f}"

                )

                print()

                print("-" * 60)

                print(
                    f"Train accuracy : {train_acc:.4f}"
                )

                print(
                    f"Video val acc  : {val['video_accuracy']:.4f}"
                )

                print(
                    f"Frame val acc  : {val['frame_accuracy']:.4f}"
                )

                print(
                    f"Precision      : {val['precision']:.4f}"
                )

                print(
                    f"Recall         : {val['recall']:.4f}"
                )

                print(
                    f"F1-score       : {val['f1']:.4f}"
                )

                print("-" * 60)

                torch.save(

                    self.model.state_dict(),

                    LAST_MODEL,

                )

                if (
                    val["accuracy"]
                    > self.best_accuracy
                ):

                    self.best_accuracy = val["accuracy"]

                    torch.save(

                        self.model.state_dict(),

                        BEST_MODEL,

                    )

                    LOGGER.info(
                        "Best model updated."
                    )

                save_checkpoint(

                    self.model,

                    self.optimizer,

                    self.scheduler,

                    epoch + 1,

                    0,

                    self.best_accuracy,

                )

                self.scheduler.step(
                    val["accuracy"]
                )

        except KeyboardInterrupt:

            LOGGER.info(
                "Training interrupted."
            )

            save_checkpoint(

                self.model,

                self.optimizer,

                self.scheduler,

                epoch,

                0,

                self.best_accuracy,

            )

            print()

            print(
                "Checkpoint saved."
            )

            print(
                "Continue next time from the same epoch."
            )
