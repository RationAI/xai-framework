from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Imagenette

from concepts.data.common import IMAGENET_MEAN, IMAGENET_STD, subsample


def build_imagenette_loader(
    root: str,
    split: str = "val",
    size: str = "160px",
    image_size: int = 224,
    batch_size: int = 64,
    num_samples: int | None = 2000,
    num_workers: int = 4,
    download: bool = True,
    seed: int = 0,
) -> "DataLoader[tuple[torch.Tensor, int]]":
    """Ungated Imagenette loader; a licensing-free substitute for ImageNet-1k."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    Path(root).mkdir(parents=True, exist_ok=True)
    dataset = Imagenette(
        root=root, split=split, size=size, download=download, transform=transform
    )
    dataset = subsample(dataset, num_samples, seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
