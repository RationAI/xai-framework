import torch
from torch import Tensor

from concepts.methods import ConceptAutoencoder
from concepts.typing import ConceptBatch, LatentBatch


def reconstruction_error(
    autoencoder: ConceptAutoencoder,
    z: LatentBatch,
    u: ConceptBatch,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    """RE(A) = mse_p(h, D o E o h), with u = E(z), streamed over chunks of samples.

    `z` may stay on cpu: each chunk and its reconstruction D(u) exist on
    `device` only for that chunk, so D(u) is never held for all of z at once
    (at layer4 it is as large as z itself).
    """
    per_sample = [
        (z_chunk.to(device) - autoencoder.decode(u_chunk.to(device)))
        .flatten(1)
        .pow(2)
        .sum(dim=1)
        for z_chunk, u_chunk in zip(
            z.split(batch_size), u.split(batch_size), strict=True
        )
    ]
    return torch.cat(per_sample).mean()
