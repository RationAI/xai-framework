import io
from bisect import bisect_right
from glob import glob
from itertools import accumulate
from typing import Any

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from concepts.data.common import IMAGENET_MEAN, IMAGENET_STD, subsample


class _ParquetImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Reads a HF-style parquet-sharded image dataset (columns: image{bytes,path}, label).

    Each shard is read and cached in full on first access to the shard, rather
    than row-by-row, since parquet doesn't support cheap single-row reads.
    """

    def __init__(self, shard_paths: list[str], transform: transforms.Compose) -> None:
        self.shard_paths = shard_paths
        self.transform = transform
        self._row_counts = [pq.ParquetFile(p).metadata.num_rows for p in shard_paths]
        self._offsets = list(accumulate([0, *self._row_counts]))
        self._shard_cache: dict[int, list[dict[str, Any]]] = {}

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        shard_idx = bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[shard_idx]

        if shard_idx not in self._shard_cache:
            table = pq.read_table(
                self.shard_paths[shard_idx], columns=["image", "label"]
            )
            self._shard_cache[shard_idx] = table.to_pylist()

        row = self._shard_cache[shard_idx][local_idx]
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        return self.transform(image), row["label"]


def build_imagenet_loader(
    root: str,
    split: str = "validation",
    image_size: int = 224,
    batch_size: int = 64,
    num_samples: int | None = 2000,
    num_workers: int = 0,
    seed: int = 0,
) -> "DataLoader[tuple[torch.Tensor, int]]":
    """Local ImageNet-1k loader over a pre-downloaded HF parquet-sharded repo.

    `num_workers` defaults to 0: each worker would build its own per-shard
    cache, and with ~350MB/shard this multiplies memory use fast.
    """
    shard_paths = sorted(glob(f"{root}/data/{split}-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(
            f"no '{split}-*.parquet' shards found under {root}/data"
        )

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    sampled = subsample(_ParquetImageDataset(shard_paths, transform), num_samples, seed)
    return DataLoader(
        sampled, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
