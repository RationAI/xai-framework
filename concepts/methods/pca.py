import torch

from concepts.methods.common import from_rows, to_rows, zero_all_but
from concepts.typing import ConceptBatch, LatentBatch


class PCAConceptAutoencoder:
    """Top-k PCA concept autoencoder; the framework's cited special case (Yeh et al., 2020).

    Works on flat [N,d] latents or spatial [N,C,H,W] ones (one concept
    representation per spatial location, as in the framework's ICE example).
    """

    def __init__(self, mean: LatentBatch, components: LatentBatch) -> None:
        self.mean = mean  # [C]
        self.components = components  # [k, C], orthonormal rows
        self.num_concepts = components.shape[0]

    def encode(self, z: LatentBatch) -> ConceptBatch:
        rows = to_rows(z)
        u = (rows - self.mean) @ self.components.T
        return from_rows(u, z)

    def decode(self, u: ConceptBatch) -> LatentBatch:
        rows = to_rows(u)
        z = rows @ self.components + self.mean
        return from_rows(z, u)

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch:
        return zero_all_but(u, index)


class PCAMethod:
    def fit(self, z: LatentBatch, num_concepts: int) -> PCAConceptAutoencoder:
        rows = to_rows(z)
        mean = rows.mean(dim=0)
        centered = rows - mean
        _, _, v = torch.pca_lowrank(centered, q=min(num_concepts, centered.shape[1]))
        components = v[:, :num_concepts].T
        return PCAConceptAutoencoder(mean=mean, components=components)
