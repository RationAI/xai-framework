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
