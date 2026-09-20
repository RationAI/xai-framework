import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from concepts.methods.common import from_rows, to_rows, zero_all_but
from concepts.typing import ConceptBatch, LatentBatch


class NonlinearAEConceptAutoencoder:
    """Autoencoder with a nonlinear decoder: gamma=g o D is no longer affine.

    Contrasts with SAE/PCA/NMF/KMeans (all affine decoders) to exercise the
    curvature term M in Theorem 2's attribution-completeness bound, instead
    of the affine special case (M=0).
    """

    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.num_concepts = decoder[0].in_features

    def encode(self, z: LatentBatch) -> ConceptBatch:
        rows = to_rows(z)
        with torch.no_grad():
            u = self.encoder(rows)
        return from_rows(u, z)

    def decode(self, u: ConceptBatch) -> LatentBatch:
        rows = to_rows(u)
        z_hat = self.decoder(rows)
        return from_rows(z_hat, u)

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch:
        return zero_all_but(u, index)


class NonlinearAEMethod:
    def __init__(
        self,
        hidden_multiplier: int = 4,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 4096,
        seed: int = 0,
    ) -> None:
        self.hidden_multiplier = hidden_multiplier
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, z: LatentBatch, num_concepts: int) -> NonlinearAEConceptAutoencoder:
        generator = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)

        rows = to_rows(z)
        d = rows.shape[1]
        hidden = min(self.hidden_multiplier * num_concepts, d)

        encoder = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_concepts),
            nn.ReLU(),
        )
        decoder = nn.Sequential(
            nn.Linear(num_concepts, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

        optimizer = torch.optim.Adam(
            [*encoder.parameters(), *decoder.parameters()], lr=self.lr
        )
        loader: DataLoader[tuple[Tensor]] = DataLoader(
            TensorDataset(rows),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                z_hat = decoder(encoder(batch))
                loss = torch.sum((z_hat - batch) ** 2, dim=-1).mean()
                loss.backward()
                optimizer.step()

        encoder.eval()
        decoder.eval()
        return NonlinearAEConceptAutoencoder(encoder=encoder, decoder=decoder)
