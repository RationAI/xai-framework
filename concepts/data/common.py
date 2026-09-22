from typing import Any

from torch import Generator, randperm
from torch.utils.data import Dataset, Subset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def subsample(
    dataset: "Dataset[Any]", num_samples: int | None, seed: int
) -> "Dataset[Any]":
    """Seeded random subset of `dataset`, or the dataset unchanged if small enough."""
    size = len(dataset)  # type: ignore[arg-type]
    if num_samples is None or num_samples >= size:
        return dataset
    generator = Generator().manual_seed(seed)
    indices = randperm(size, generator=generator)[:num_samples].tolist()
    # Sorted so a shuffle=False loader visits chunked/sharded datasets (e.g.
    # the parquet-backed imagenet loader) in on-disk order -- selection stays
    # a uniform random subsample, only the visiting order changes, but that
    # keeps per-shard access grouped instead of scattered across every shard.
    return Subset(dataset, sorted(indices))
