import torch
from torch import Tensor

from concepts.decomposition import TorchvisionDecomposition
from concepts.methods import ConceptAutoencoder
from concepts.metrics.common import decoded_head, decoded_head_gradient
from concepts.typing import ConceptBatch


ATTRIBUTION_RULES = ("insertion", "occlusion", "gradxinput")


def total_attributions(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    u: ConceptBatch,
    target_index: Tensor,
) -> dict[str, Tensor]:
    """Pointwise TA(A,phi;x) = b + sum_i phi(x;i) for each rule in `ATTRIBUTION_RULES`.

    gamma = g o D, b = gamma(0), C(x) = u and S_i = `autoencoder.zero_concept`:
      insertion:  phi(x;i) = gamma(S_i u) - b
      occlusion:  phi(x;i) = gamma(u) - gamma(u - S_i u)
      gradxinput: phi(x;i) = <grad gamma(u), S_i u>

    `target_index[n]` picks the scalar output coordinate explained for sample n
    (the framework's attribution setting has N=1) -- typically the predicted
    class. Returns shape [N] per rule.

    ATE compares these against f(x), ADD against f_A(x) (both at target_index).
    """
    n = u.shape[0]
    rows = torch.arange(n, device=u.device)

    # no_grad: fixed forward passes repeated num_concepts times, not a gradient
    # computation -- decode/g otherwise build a needless backward graph whenever
    # the method has learnable decoder weights (SAE/NonlinearAE). The gradient
    # for gradxinput is taken separately, with its own chunked graph.
    with torch.no_grad():
        zero = torch.zeros_like(u[:1])
        baseline = decoded_head(decomposition, autoencoder, zero)[0][
            target_index
        ]  # [N]
        full = decoded_head(decomposition, autoencoder, u)[rows, target_index]

        insertion = baseline.clone()
        occlusion = baseline.clone()
        for i in range(autoencoder.num_concepts):
            isolated = autoencoder.zero_concept(u, i)
            inserted = decoded_head(decomposition, autoencoder, isolated)
            removed = decoded_head(decomposition, autoencoder, u - isolated)
            insertion = insertion + (inserted[rows, target_index] - baseline)
            occlusion = occlusion + (full - removed[rows, target_index])

    grad = decoded_head_gradient(decomposition, autoencoder, u, target_index)
    gradxinput = baseline.clone()
    for i in range(autoencoder.num_concepts):
        isolated = autoencoder.zero_concept(u, i)
        gradxinput = gradxinput + (grad * isolated).flatten(1).sum(dim=1)

    return {
        "insertion": insertion,
        "occlusion": occlusion,
        "gradxinput": gradxinput,
    }


def representation_fourth_moment_root(u: ConceptBatch) -> Tensor:
    """sqrt(E_x[||C(x)||_2^4]), the factor multiplying M in the ADD bound (Theorem 2)."""
    return u.flatten(1).pow(2).sum(dim=1).pow(2).mean().sqrt()
