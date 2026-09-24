import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn


def train_with_early_stopping(
    module: nn.Module,
    forward: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    targets: Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    val_fraction: float,
    patience: int,
    seed: int,
) -> None:
    """Fits `module` so `forward(*inputs)` matches `targets`, restoring its best-val state.

    Loss is E[||prediction - target||_2^2], the same MSE the metrics use. The
    validation rows are a random `val_fraction` of the given rows (the fit split
    is in dataset order, which need not be shuffled). The state before any
    training counts as a checkpoint, so a head initialized at gamma = g o D can
    never end up with a worse validation loss than g o D itself.
    """
    generator = torch.Generator().manual_seed(seed)
    n = targets.shape[0]
    order = torch.randperm(n, generator=generator).to(targets.device)
    num_val = max(1, int(val_fraction * n))
    val_idx, train_idx = order[:num_val], order[num_val:]

    def val_loss() -> float:
        module.eval()
        with torch.no_grad():
            total = 0.0
            for chunk in val_idx.split(batch_size):
                prediction = forward(*(x[chunk] for x in inputs))
                total += torch.sum((prediction - targets[chunk]) ** 2).item()
        module.train()
        return total / num_val

    optimizer = torch.optim.Adam(
        [p for p in module.parameters() if p.requires_grad], lr=lr
    )
    best_loss = val_loss()
    best_state = copy.deepcopy(module.state_dict())
    epochs_without_improvement = 0
    for _ in range(epochs):
        shuffled = train_idx[
            torch.randperm(train_idx.shape[0], generator=generator).to(targets.device)
        ]
        for chunk in shuffled.split(batch_size):
            optimizer.zero_grad()
            prediction = forward(*(x[chunk] for x in inputs))
            loss = torch.mean(torch.sum((prediction - targets[chunk]) ** 2, dim=-1))
            loss.backward()
            optimizer.step()

        epoch_val_loss = val_loss()
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_state = copy.deepcopy(module.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    module.load_state_dict(best_state)
    module.eval()
