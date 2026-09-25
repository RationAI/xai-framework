import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from concepts.methods.common import apply_batched, num_epochs
from concepts.typing import ConceptBatch, OutputBatch


class MLPProbe:
    """Learned gamma: V -> Y standing in for the inf over gamma in MCE (an upper bound, not the true infimum)."""

    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 50,
        min_steps: int = 5000,
        lr: float = 1e-3,
        batch_size: int = 4096,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.min_steps = min_steps
        self.lr = lr
        self.batch_size = batch_size
        self.net: nn.Sequential | None = None

    def fit(self, u: ConceptBatch, targets: OutputBatch) -> "MLPProbe":
        u = u.flatten(1)  # gamma takes the whole representation u, spatial or not
        self.net = nn.Sequential(
            nn.Linear(u.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, targets.shape[1]),
        ).to(u.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            TensorDataset(u, targets),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        epochs = num_epochs(self.epochs, self.min_steps, u.shape[0], self.batch_size)
        for _ in range(epochs):
            for batch_u, batch_targets in loader:
                optimizer.zero_grad()
                loss = torch.mean(
                    torch.sum((self.net(batch_u) - batch_targets) ** 2, dim=-1)
                )
                loss.backward()
                optimizer.step()
        return self

    def __call__(self, u: ConceptBatch) -> Tensor:
        assert self.net is not None, "MLPProbe.fit() must be called before use"
        with torch.no_grad():
            return apply_batched(self.net, u.flatten(1), self.batch_size)
