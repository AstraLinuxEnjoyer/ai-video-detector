from training.data.catalog import ensure_catalog

__all__ = ["build_datasets", "ensure_catalog"]


def build_datasets(*args, **kwargs):
    from training.data.pipeline import build_datasets as _build_datasets
    return _build_datasets(*args, **kwargs)
