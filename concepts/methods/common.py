import math
from collections.abc import Callable

import torch
from torch import Tensor


def apply_batched(
    fn: Callable[[Tensor], Tensor],
    rows: Tensor,
    batch_size: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Applies `fn` to `rows` in chunks along dim 0, concatenating the results.

    Bounds peak memory for row counts that can be far larger than the number
    of samples (`to_rows` turns spatial [N,C,H,W] latents into [N*H*W, C]).
    Uses `torch.split`/`torch.cat`, so gradients still flow through to `rows`
    when `fn` is used inside an autograd graph (see `estimate_gamma_curvature`).

    With `device`, each chunk is moved there before `fn` and its result moved
    back to `rows.device`, so `rows` can stay on cpu (e.g. latents too large
    for gpu memory) while `fn` runs on the gpu one chunk at a time.
    """

    def run(chunk: Tensor) -> Tensor:
        if device is None:
            return fn(chunk)
        return fn(chunk.to(device)).to(rows.device)

    if rows.shape[0] <= batch_size:
        return run(rows)
    return torch.cat([run(chunk) for chunk in rows.split(batch_size)], dim=0)


def num_epochs(epochs: int, min_steps: int, num_rows: int, batch_size: int) -> int:
    """`epochs`, raised if needed so training takes at least `min_steps` optimizer steps.

    An epoch is one step per batch, so the same `epochs` gives very different
    budgets across boundaries: spatial latents yield H*W rows per sample
    (layer4: 49), a flat latent only one. E.g. 12,500 penultimate rows at batch
    size 4096 are 4 steps per epoch, so 50 epochs are only 200 steps.
    """
    steps_per_epoch = math.ceil(num_rows / batch_size)
    return max(epochs, math.ceil(min_steps / steps_per_epoch))


def to_rows(z: Tensor) -> Tensor:
    """[N,d] unchanged, or spatial [N,C,H,W] -> per-location channel rows [N*H*W, C]."""
    if z.dim() == 2:
        return z
    n, c, h, w = z.shape
    return z.permute(0, 2, 3, 1).reshape(n * h * w, c)


def from_rows(rows: Tensor, like: Tensor) -> Tensor:
    """Inverse of `to_rows`: reshape row-wise outputs back to `like`'s batch/spatial layout."""
    if like.dim() == 2:
        return rows
    n, _, h, w = like.shape
    return rows.reshape(n, h, w, rows.shape[-1]).permute(0, 3, 1, 2)


def zero_all_but(u: Tensor, index: int) -> Tensor:
    """S_i from the framework: zero every concept block except `index`."""
    masked = torch.zeros_like(u)
    masked[:, index] = u[:, index]
    return masked
