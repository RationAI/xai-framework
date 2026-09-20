from torch import Tensor


def squared_l2_mse(a: Tensor, b: Tensor) -> Tensor:
    """mse_p(f1,f2)=E[||f1(x)-f2(x)||_2^2] over all non-batch dims; not torch's elementwise-mean MSE."""
    return (a - b).flatten(1).pow(2).sum(dim=1).mean()
