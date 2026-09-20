from torch import Tensor

from concepts.metrics.common import squared_l2_mse


def fidelity_error(f_x: Tensor, f_a_x: Tensor) -> Tensor:
    """FE(A) = mse_p(f, f_A), with f_A = g o D o E o h."""
    return squared_l2_mse(f_x, f_a_x)
