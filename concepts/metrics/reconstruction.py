from torch import Tensor

from concepts.metrics.common import squared_l2_mse


def reconstruction_error(z: Tensor, z_hat: Tensor) -> Tensor:
    """RE(A) = mse_p(h, D o E o h)."""
    return squared_l2_mse(z, z_hat)
