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
    total = len(loader.dataset)  # type: ignore[arg-type]
    z = logits = labels = None
    offset = 0
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="extracting latents"):
            images = images.to(device)
            batch_z = decomposition.extract_latents(images)
            batch_logits = decomposition.predict_from_latent(batch_z)
            if z is None:
                # allocated once total/shapes are known, and filled in place --
                # avoids the list-of-batches + torch.cat pattern, which briefly
                # needs both the full list AND a freshly concatenated copy
                # alive at once (2x peak on top of the already-large z, e.g.
                # ~20GB each way for a full-split run at a spatial boundary)
                z = torch.empty((total, *batch_z.shape[1:]), dtype=batch_z.dtype)
                logits = torch.empty(
                    (total, *batch_logits.shape[1:]), dtype=batch_logits.dtype
                )
                labels = torch.empty((total, *batch_labels.shape[1:]), dtype=batch_labels.dtype)
            n = batch_z.shape[0]
            z[offset : offset + n] = batch_z.cpu()
            logits[offset : offset + n] = batch_logits.cpu()
            labels[offset : offset + n] = batch_labels
            offset += n

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z, "logits": logits, "labels": labels}, cache_path)
    return z, logits, labels
