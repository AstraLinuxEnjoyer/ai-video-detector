from pathlib import Path
import logging
import csv
import torch


class TrainingManager:

    def __init__(self, cache_dir: Path):

        self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.best_model = self.cache_dir / "best.pt"
        self.last_model = self.cache_dir / "last.pt"

        self.checkpoint = self.cache_dir / "checkpoint.pt"

        self.metrics_csv = self.cache_dir / "metrics.csv"

        self.log_file = self.cache_dir / "training.log"

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            force=True
        )

        if not self.metrics_csv.exists():

            with open(self.metrics_csv, "w", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_acc",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "learning_rate"
                ])

    def log(self, message: str):

        logging.info(message)

    def save_metrics(
            self,
            epoch,
            train_loss,
            train_acc,
            val_acc,
            precision,
            recall,
            f1,
            roc_auc,
            lr
    ):

        with open(
                self.metrics_csv,
                "a",
                newline="",
                encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(train_acc, 6),
                round(val_acc, 6),
                round(precision, 6),
                round(recall, 6),
                round(f1, 6),
                round(roc_auc, 6),
                lr
            ])

    def save_checkpoint(

            self,

            epoch,

            model,

            optimizer,

            scheduler,

            best_acc

    ):

        torch.save(

            {

                "epoch": epoch,

                "model_state": model.state_dict(),

                "optimizer_state": optimizer.state_dict(),

                "scheduler_state":
                    scheduler.state_dict()
                    if scheduler is not None
                    else None,

                "best_acc": best_acc

            },

            self.checkpoint

        )

        logging.info(
            f"Checkpoint saved. Epoch={epoch}"
        )

    def load_checkpoint(

            self,

            model,

            optimizer=None,

            scheduler=None,

            device="cpu"

    ):

        if not self.checkpoint.exists():

            return 0, 0.0

        checkpoint = torch.load(

            self.checkpoint,

            map_location=device

        )

        model.load_state_dict(

            checkpoint["model_state"]

        )

        if optimizer is not None:

            optimizer.load_state_dict(

                checkpoint["optimizer_state"]

            )

        if (

            scheduler is not None

            and checkpoint["scheduler_state"] is not None

        ):

            scheduler.load_state_dict(

                checkpoint["scheduler_state"]

            )

        logging.info(

            f"Checkpoint loaded. Epoch={checkpoint['epoch']}"

        )

        return (

            checkpoint["epoch"],

            checkpoint["best_acc"]

        )

    def save_best(

            self,

            model

    ):

        torch.save(

            model.state_dict(),

            self.best_model

        )

        logging.info(

            "Saved BEST model"

        )

    def save_last(

            self,

            model

    ):

        torch.save(

            model.state_dict(),

            self.last_model

        )

    def save_interrupt(

            self,

            epoch,

            model,

            optimizer,

            scheduler,

            best_acc

    ):

        print()

        print("=" * 60)

        print("Training interrupted.")

        print("Saving checkpoint...")

        print("=" * 60)

        print()

        self.save_checkpoint(

            epoch,

            model,

            optimizer,

            scheduler,

            best_acc

        )

        self.save_last(model)

        print("Checkpoint successfully saved.")

        print()

    def has_checkpoint(self):

        return self.checkpoint.exists()