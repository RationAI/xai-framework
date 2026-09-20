import torch
from torch import Tensor, nn

from concepts.typing import ConceptBatch, OutputBatch


class MLPProbe:
    """Learned gamma: V -> Y standing in for the inf over gamma in MCE (an upper bound, not the true infimum)."""

    def __init__(
        self, hidden_dim: int = 128, epochs: int = 50, lr: float = 1e-3
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.net: nn.Sequential | None = None

    def fit(self, u: ConceptBatch, targets: OutputBatch) -> "MLPProbe":
        u = u.flatten(1)  # gamma takes the whole representation u, spatial or not
        self.net = nn.Sequential(
            nn.Linear(u.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, targets.shape[1]),
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = torch.mean(torch.sum((self.net(u) - targets) ** 2, dim=-1))
            loss.backward()
            optimizer.step()
        return self

    def __call__(self, u: ConceptBatch) -> Tensor:
        assert self.net is not None, "MLPProbe.fit() must be called before use"
        with torch.no_grad():
            return self.net(u.flatten(1))
