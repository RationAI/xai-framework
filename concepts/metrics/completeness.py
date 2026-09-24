from torch import Tensor

from concepts.metrics.common import squared_l2_mse


def model_completeness_error(probe_predictions: Tensor, targets: Tensor) -> Tensor:
    """MCE_UB: a fitted probe's error, an upper bound on MCE(A) = inf_gamma mse_p(f, gamma o E o h)."""
    return squared_l2_mse(probe_predictions, targets)
