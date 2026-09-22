import torch

from concepts.methods.common import apply_batched, from_rows, to_rows, zero_all_but
from concepts.typing import ConceptBatch, LatentBatch


class PCAConceptAutoencoder:
    """Top-k PCA concept autoencoder; the framework's cited special case (Yeh et al., 2020).

    Works on flat [N,d] latents or spatial [N,C,H,W] ones (one concept
    representation per spatial location, as in the framework's ICE example).
    """

    def __init__(
        self, mean: LatentBatch, components: LatentBatch, batch_size: int = 4096
    ) -> None:
        self.mean = mean  # [C]
        self.components = components  # [k, C], orthonormal rows
        self.num_concepts = components.shape[0]
        self.batch_size = batch_size

    def encode(self, z: LatentBatch) -> ConceptBatch:
        rows = to_rows(z)
        u = apply_batched(
            lambda r: (r - self.mean) @ self.components.T, rows, self.batch_size
        )
        return from_rows(u, z)

    def decode(self, u: ConceptBatch) -> LatentBatch:
        rows = to_rows(u)
        z = apply_batched(
            lambda r: r @ self.components + self.mean, rows, self.batch_size
        )
        return from_rows(z, u)

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch:
        return zero_all_but(u, index)


class PCAMethod:
    def __init__(self, batch_size: int = 4096) -> None:
        self.batch_size = batch_size

    def fit(self, z: LatentBatch, num_concepts: int) -> PCAConceptAutoencoder:
        rows = to_rows(z)
        mean = rows.mean(dim=0)
        centered = rows - mean
        del rows
        _, _, v = torch.pca_lowrank(
            centered, q=min(num_concepts, centered.shape[1]), center=False
        )
        components = v[:, :num_concepts].T
        return PCAConceptAutoencoder(
            mean=mean, components=components, batch_size=self.batch_size
        )
