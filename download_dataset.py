"""
Legacy helper — use the automatic pipeline instead:

  python training/prepare_dataset.py   # fetch catalog + split manifests
  python -m training.train_spatial     # download videos on demand + train
"""

from training.data.catalog import ensure_catalog
from training.data.pipeline import prepare_training_manifests


def main():
    print("Note: manual bulk download is no longer required.")
    print("Building remote catalog and train/val manifests...\n")

    ensure_catalog(refresh=True)
    train_entries, val_entries = prepare_training_manifests(refresh_catalog=True)

    print(f"Ready: {len(train_entries)} train / {len(val_entries)} val videos")
    print("Start training with: python -m training.train_spatial")


if __name__ == "__main__":
    main()
