import torch
from torch import Tensor, nn

from concepts.methods.common import apply_batched
from concepts.probes.training import train_with_early_stopping
from concepts.typing import ConceptBatch, OutputBatch


class ResidualProbe:
    """gamma(u) = g(D(u)) + r(u): a learned correction on top of the decoded head.

    `r` is an MLP on the normalized, flattened u whose last layer starts at
    zero, so gamma starts exactly at g o D (i.e. at FE) and training can only
    improve on it (see train_with_early_stopping). g(D(u)) is passed in
    precomputed as `base`, since g and D stay fixed here.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_fraction: float = 0.1,
        patience: int = 10,
        seed: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.patience = patience
        self.seed = seed
        self.net: nn.Sequential | None = None
        self.mean: Tensor | None = None
        self.std: Tensor | None = None

    def _standardize(self, u: ConceptBatch) -> Tensor:
        assert self.mean is not None and self.std is not None
        return (u.flatten(1) - self.mean) / self.std

    def fit(
        self, u: ConceptBatch, base: OutputBatch, targets: OutputBatch
    ) -> "ResidualProbe":
        torch.manual_seed(self.seed)
        flat = u.flatten(1)
        self.mean = flat.mean(dim=0)
        # One global scale, not per-feature: near-constant features (dead SAE
        # units, near one-hot KMeans memberships) would otherwise be divided by
        # a ~0 std and blow up on eval rows where they are nonzero.
        self.std = flat.std().clamp_min(1e-8)
        output = nn.Linear(self.hidden_dim, targets.shape[1])
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.net = nn.Sequential(
            nn.Linear(flat.shape[1], self.hidden_dim), nn.ReLU(), output
        ).to(u.device)

        net = self.net
        train_with_early_stopping(
            net,
            lambda x, b: b + net(x),
            (self._standardize(u), base),
            targets,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            val_fraction=self.val_fraction,
            patience=self.patience,
            seed=self.seed,
        )
        return self

    def __call__(self, u: ConceptBatch, base: OutputBatch) -> Tensor:
        assert self.net is not None, "ResidualProbe.fit() must be called before use"
        with torch.no_grad():
            return base + apply_batched(self.net, self._standardize(u), self.batch_size)
