import copy

import torch
from torch import Tensor, nn

from concepts.methods.common import apply_batched, from_rows, to_rows
from concepts.probes.training import train_with_early_stopping
from concepts.typing import ConceptBatch, OutputBatch


def _pool(u: ConceptBatch) -> Tensor:
    """Spatial [N,k,H,W] -> per-concept spatial mean [N,k]; flat [N,k] unchanged."""
    return u if u.dim() == 2 else u.mean(dim=(2, 3))


class AffineDecodedHead:
    """Best head of the form g' o D' for an affine D and affine g, fit in closed form.

    At the penultimate and layer4 boundaries g is affine (fc, or avgpool + fc),
    and D acts affinely on each spatial row, so g o D(u) = W D u_bar + c with
    u_bar the spatial mean of u. Refitting W, D and c jointly therefore spans
    exactly the affine maps of u_bar, whose optimum is ordinary least squares.
    This family contains g o D itself, so on the fit rows it is never worse.
    """

    def __init__(self) -> None:
        self.weight: Tensor | None = None  # [k + 1, out_dim], last row = intercept

    @staticmethod
    def _design(u: ConceptBatch) -> Tensor:
        pooled = _pool(u).double().cpu()
        return torch.cat([pooled, torch.ones(pooled.shape[0], 1).double()], dim=1)

    def fit(self, u: ConceptBatch, targets: OutputBatch) -> "AffineDecodedHead":
        # gelsd tolerates rank-deficient designs, e.g. KMeans memberships sum to 1
        # and so are collinear with the intercept column; CUDA lstsq does not.
        solution = torch.linalg.lstsq(
            self._design(u), targets.double().cpu(), driver="gelsd"
        )
        self.weight = solution.solution
        return self

    def __call__(self, u: ConceptBatch) -> Tensor:
        assert self.weight is not None, (
            "AffineDecodedHead.fit() must be called before use"
        )
        return (self._design(u) @ self.weight).to(u)


class _DecodedHead(nn.Module):
    def __init__(self, decoder: nn.Module, g: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.g = g

    def forward(self, u: Tensor) -> Tensor:
        return self.g(from_rows(self.decoder(to_rows(u)), u))


class FineTunedDecodedHead:
    """Head g' o D' for a nonlinear D: copies of D and g, fine-tuned from their weights.

    There is no closed form once D is nonlinear, so both copies are trained by
    gradient descent, starting exactly at g o D (FE). The originals are left
    untouched, since FE, ATE and ADD still use them.
    """

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 1e-4,
        batch_size: int = 256,
        val_fraction: float = 0.1,
        patience: int = 10,
        seed: int = 0,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.patience = patience
        self.seed = seed
        self.head: _DecodedHead | None = None

    def fit(
        self, u: ConceptBatch, targets: OutputBatch, decoder: nn.Module, g: nn.Module
    ) -> "FineTunedDecodedHead":
        torch.manual_seed(self.seed)
        head = _DecodedHead(copy.deepcopy(decoder), copy.deepcopy(g)).to(u.device)
        for param in head.parameters():
            param.requires_grad_(True)
        train_with_early_stopping(
            head,
            head,
            (u,),
            targets,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            val_fraction=self.val_fraction,
            patience=self.patience,
            seed=self.seed,
        )
        self.head = head
        return self

    def __call__(self, u: ConceptBatch) -> Tensor:
        assert self.head is not None, (
            "FineTunedDecodedHead.fit() must be called before use"
        )
        with torch.no_grad():
            return apply_batched(self.head, u, self.batch_size)
