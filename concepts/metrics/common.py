import torch
from torch import Tensor

from concepts.decomposition import TorchvisionDecomposition
from concepts.methods import ConceptAutoencoder
from concepts.methods.common import apply_batched
from concepts.typing import ConceptBatch, OutputBatch


def squared_l2_mse(a: Tensor, b: Tensor) -> Tensor:
    """mse_p(f1,f2)=E[||f1(x)-f2(x)||_2^2] over all non-batch dims; not torch's elementwise-mean MSE."""
    return (a - b).flatten(1).pow(2).sum(dim=1).mean()


def decoded_head(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    u: ConceptBatch,
) -> OutputBatch:
    """gamma(u) = g(D(u)), chunked over samples on the decomposition's device.

    D(u) is latent-sized (as large as z at layer4), so it only ever exists for
    one chunk at a time; the outputs come back on `u`'s device. Stays
    autograd-compatible (split/cat), like `apply_batched`.
    """
    return apply_batched(
        lambda chunk: decomposition.g(autoencoder.decode(chunk)),
        u,
        decomposition.batch_size,
        device=decomposition.device,
    )


def decoded_head_gradient(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    v: Tensor,
    target_index: Tensor,
) -> Tensor:
    """Per-row gradient of gamma_i = P_{target_index[i]} o g o D at v[i].

    Rows are independent, so this runs one chunk of samples at a time on the
    decomposition's device (one small autograd graph per chunk, instead of a
    graph through D(v) for every row at once) and returns on `v`'s device.
    """
    device = decomposition.device
    grads = []
    for v_chunk, target_chunk in zip(
        v.split(decomposition.batch_size),
        target_index.split(decomposition.batch_size),
        strict=True,
    ):
        v_chunk = v_chunk.detach().to(device).requires_grad_(True)
        out = decomposition.g(autoencoder.decode(v_chunk))
        rows = torch.arange(v_chunk.shape[0], device=device)
        scalar_out = out[rows, target_chunk.to(device)]
        (grad_v,) = torch.autograd.grad(scalar_out.sum(), v_chunk)
        grads.append(grad_v.detach().to(v.device))
    return torch.cat(grads)
