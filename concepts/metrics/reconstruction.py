from collections.abc import Iterator

import torch
from torch import Tensor

from concepts.methods import ConceptAutoencoder
from concepts.typing import ConceptBatch, LatentBatch


def _residual_chunks(
    autoencoder: ConceptAutoencoder,
    z: LatentBatch,
    u: ConceptBatch,
    batch_size: int,
    device: torch.device | str,
) -> Iterator[Tensor]:
    """Residuals e = z - D(u), one chunk of samples at a time, on `device`.

    `z` may stay on cpu: each chunk and its reconstruction D(u) exist on
    `device` only for that chunk, so D(u) is never held for all of z at once
    (at layer4 it is as large as z itself).
    """
    for z_chunk, u_chunk in zip(z.split(batch_size), u.split(batch_size), strict=True):
        yield z_chunk.to(device) - autoencoder.decode(u_chunk.to(device))


def reconstruction_error(
    autoencoder: ConceptAutoencoder,
    z: LatentBatch,
    u: ConceptBatch,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    """RE(A) = mse_p(h, D o E o h) = E||e||^2, with u = E(z), streamed over chunks of samples."""
    per_sample = [
        e.flatten(1).pow(2).sum(dim=1)
        for e in _residual_chunks(autoencoder, z, u, batch_size, device)
    ]
    return torch.cat(per_sample).mean()


def pooled_reconstruction_error(
    autoencoder: ConceptAutoencoder,
    z: LatentBatch,
    u: ConceptBatch,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    """E||e_bar||^2, where e_bar is the residual e averaged over spatial locations.

    At layer4 and penultimate g(z) = W z_bar + c, so f - f_A = W e_bar; this is
    the quantity behind the alignment/spatial-coherence split of rho_FE
    (eq. cg-fidelity-slack). For a flat latent e_bar = e and this equals RE.
    """
    per_sample = [
        e.flatten(2).mean(dim=2).pow(2).sum(dim=1) if e.dim() > 2 else e.pow(2).sum(1)
        for e in _residual_chunks(autoencoder, z, u, batch_size, device)
    ]
    return torch.cat(per_sample).mean()
