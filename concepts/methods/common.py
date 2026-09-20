import torch
from torch import Tensor


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
