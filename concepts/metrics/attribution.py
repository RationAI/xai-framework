import torch
from torch import Tensor

from concepts.decomposition import TorchvisionDecomposition
from concepts.methods import ConceptAutoencoder
from concepts.typing import ConceptBatch, OutputBatch


def attribution_error(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    f_x: OutputBatch,
    u: ConceptBatch,
    target_index: Tensor,
) -> Tensor:
    """Pointwise AE(A;x) = |f(x) - gamma(0) - sum_i phi(x;i)|, gamma = g o D.

    `target_index[n]` picks the scalar output coordinate explained for sample n
    (the framework's f is scalar-valued) -- typically the predicted class.
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

        contributions = torch.zeros(n, device=u.device)
        for i in range(autoencoder.num_concepts):
            u_i = autoencoder.zero_concept(u, i)
            gamma_i_full = decomposition.predict_from_latent(
                autoencoder.decode(u_i)
            )  # [N, out_dim]
            contributions = contributions + (gamma_i_full[rows, target_index] - baseline)

    f_r = f_x[rows, target_index]
    return (f_r - baseline - contributions).abs()


def mean_squared_attribution_error(pointwise_ae: Tensor) -> Tensor:
    """AE_p(A) = E_x[AE(A;x)^2]."""
    return (pointwise_ae**2).mean()
