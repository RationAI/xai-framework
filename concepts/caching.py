import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from concepts.decomposition import TorchvisionDecomposition
from concepts.typing import LatentBatch, OutputBatch


def _cache_key(
    model_name: str, feature_node: str, data_name: str, num_samples: int | None
) -> str:
    raw = f"{model_name}|{feature_node}|{data_name}|{num_samples}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def extract_and_cache_latents(
    decomposition: TorchvisionDecomposition,
    loader: "DataLoader[tuple[torch.Tensor, int]]",
    cache_dir: str,
    data_name: str,
    num_samples: int | None,
    device: str = "cpu",
) -> tuple[LatentBatch, OutputBatch, torch.Tensor]:
    """Extracts (z, f(x), labels) once and caches them under `cache_dir`, kept forever."""
    key = _cache_key(
        decomposition.name, decomposition.feature_node, data_name, num_samples
    )
    cache_path = Path(cache_dir) / f"latents_{key}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path)
        return cached["z"], cached["logits"], cached["labels"]

    decomposition = decomposition.to(device)
    all_z, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="extracting latents"):
            images = images.to(device)
            z = decomposition.extract_latents(images)
            logits = decomposition.predict_from_latent(z)
            all_z.append(z.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    z = torch.cat(all_z)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z, "logits": logits, "labels": labels}, cache_path)
    return z, logits, labels
