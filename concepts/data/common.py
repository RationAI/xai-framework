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
    return Subset(dataset, indices)
