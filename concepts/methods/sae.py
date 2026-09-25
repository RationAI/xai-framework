from typing import Self

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from concepts.methods.common import (
    apply_batched,
    from_rows,
    num_epochs,
    to_rows,
    zero_all_but,
)
from concepts.typing import ConceptBatch, LatentBatch


class SAEConceptAutoencoder:
    """Sparse autoencoder: 2-layer Linear+ReLU encoder, single affine decoder.

    The encoder's final ReLU gives nonnegative, sparsity-encouraged concept
    codes; the decoder stays affine so gamma=g o D keeps the curvature-free
    (M=0) case of the attribution bound (Theorem 2), unlike NonlinearAE.
    """

    def __init__(
        self, encoder: nn.Sequential, decoder: nn.Linear, batch_size: int = 4096
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.num_concepts = decoder.in_features
        self.batch_size = batch_size
        self.affine_decoder = True

    def encode(self, z: LatentBatch) -> ConceptBatch:
        rows = to_rows(z)
        with torch.no_grad():
            u = apply_batched(self.encoder, rows, self.batch_size)
        return from_rows(u, z)

    def decode(self, u: ConceptBatch) -> LatentBatch:
        rows = to_rows(u)
        z_hat = apply_batched(self.decoder, rows, self.batch_size)
        return from_rows(z_hat, u)

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch:
        return zero_all_but(u, index)

    def to(self, device: torch.device | str) -> Self:
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return self


class SAEMethod:
    def __init__(
        self,
        hidden_multiplier: int = 4,
        sparsity_weight: float = 1e-3,
        epochs: int = 50,
        min_steps: int = 5000,
        lr: float = 1e-3,
        batch_size: int = 4096,
        seed: int = 0,
    ) -> None:
        self.hidden_multiplier = hidden_multiplier
        self.sparsity_weight = sparsity_weight
        self.epochs = epochs
        self.min_steps = min_steps
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def fit(
        self,
        z: LatentBatch,
        num_concepts: int,
        device: torch.device | str | None = None,
    ) -> SAEConceptAutoencoder:
        generator = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)

        # rows stay where z is (possibly cpu); only mini-batches go to `device`
        device = device or z.device
        rows = to_rows(z)
        d = rows.shape[1]
        hidden = min(self.hidden_multiplier * num_concepts, d)

        encoder = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_concepts),
            nn.ReLU(),
        ).to(device)
        decoder = nn.Linear(num_concepts, d).to(device)

        optimizer = torch.optim.Adam(
            [*encoder.parameters(), *decoder.parameters()], lr=self.lr
        )
        loader: DataLoader[tuple[Tensor]] = DataLoader(
            TensorDataset(rows),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        epochs = num_epochs(self.epochs, self.min_steps, rows.shape[0], self.batch_size)
        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                u = encoder(batch)
                z_hat = decoder(u)
                recon = torch.sum((z_hat - batch) ** 2, dim=-1).mean()
                sparsity = u.abs().sum(dim=-1).mean()
                loss = recon + self.sparsity_weight * sparsity
                loss.backward()
                optimizer.step()

        encoder.eval()
        decoder.eval()
        return SAEConceptAutoencoder(
            encoder=encoder, decoder=decoder, batch_size=self.batch_size
        )
