import torch
from torch import Tensor

from concepts.decomposition import TorchvisionDecomposition
from concepts.methods import ConceptAutoencoder
from concepts.typing import ConceptBatch


def insertion_total_attribution(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    u: ConceptBatch,
    target_index: Tensor,
) -> Tensor:
    """Pointwise TA(A,phi;x) = b + sum_i phi(x;i) with insertion phi(x;i) = gamma(S_i C(x)) - b.

    gamma = g o D and b = gamma(0). `target_index[n]` picks the scalar output
    coordinate explained for sample n (the framework's attribution setting has
    N=1) -- typically the predicted class. Returns shape [N].

    ATE compares this against f(x), ADD against f_A(x) (both at target_index).
    """
    n = u.shape[0]
    rows = torch.arange(n, device=u.device)

    # no_grad: this is a fixed forward pass repeated num_concepts times, not a
    # gradient computation (contrast estimate_gamma_curvature) -- decode/
    # predict_from_latent otherwise build a needless backward graph through g
    # whenever the method has learnable decoder weights (SAE/NonlinearAE).
    with torch.no_grad():
        zero = torch.zeros_like(u[:1])
        baseline_full = decomposition.predict_from_latent(autoencoder.decode(zero))[
            0
        ]  # [out_dim]
        baseline = baseline_full[target_index]  # [N]

        total = baseline.clone()
        for i in range(autoencoder.num_concepts):
            u_i = autoencoder.zero_concept(u, i)
            gamma_i_full = decomposition.predict_from_latent(
                autoencoder.decode(u_i)
            )  # [N, out_dim]
            total = total + (gamma_i_full[rows, target_index] - baseline)
    return total


def representation_fourth_moment_root(u: ConceptBatch) -> Tensor:
    """sqrt(E_x[||C(x)||_2^4]), the factor multiplying M in the ADD bound (Theorem 2)."""
    return u.flatten(1).pow(2).sum(dim=1).pow(2).mean().sqrt()
