import torch
from sklearn.cluster import KMeans as SKKMeans

from concepts.methods.common import apply_batched, from_rows, to_rows, zero_all_but
from concepts.typing import ConceptBatch, LatentBatch


class KMeansConceptAutoencoder:
    """Concept i's activation is the soft assignment to centroid i; decode is linear.

    Activations are the posterior of an isotropic equal-weight Gaussian mixture
    centred on the centroids (softmax over -||z - c_i||^2 / 2 sigma^2), so they
    sum to 1 and decode is a convex combination of centroids. Normalizing here
    rather than in decode keeps concept i's contribution u_i * c_i, which is what
    zero_concept-based attribution relies on.

    Unlike PCA/NMF, this reconstruction isn't fit to minimize RE directly, so
    it's a useful contrast baseline rather than a strong one.
    """

    def __init__(
        self, centroids: LatentBatch, bandwidth: float, batch_size: int = 4096
    ) -> None:
        self.centroids = centroids  # [k, C]
        self.bandwidth = bandwidth
        self.num_concepts = centroids.shape[0]
        self.batch_size = batch_size
        self.affine_decoder = True

    def encode(self, z: LatentBatch) -> ConceptBatch:
        rows = to_rows(z)

        def _soft_assign(chunk: torch.Tensor) -> torch.Tensor:
            dist_sq = torch.cdist(chunk, self.centroids) ** 2
            return torch.softmax(-dist_sq / (2 * self.bandwidth**2), dim=-1)

        u = apply_batched(_soft_assign, rows, self.batch_size)
        return from_rows(u, z)

    def decode(self, u: ConceptBatch) -> LatentBatch:
        rows = to_rows(u)
        z = apply_batched(lambda r: r @ self.centroids, rows, self.batch_size)
        return from_rows(z, u)

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch:
        return zero_all_but(u, index)


class KMeansMethod:
    def __init__(self, seed: int = 0, batch_size: int = 4096) -> None:
        self.seed = seed
        self.batch_size = batch_size

    def fit(self, z: LatentBatch, num_concepts: int) -> KMeansConceptAutoencoder:
        rows = to_rows(z)
        model = SKKMeans(n_clusters=num_concepts, n_init="auto", random_state=self.seed)
        model.fit(rows.detach().cpu().numpy())
        centroids = torch.from_numpy(model.cluster_centers_).to(rows)
        # Per-dimension within-cluster std (isotropic Gaussian sigma), not the
        # total distance to the centroid, which would be sqrt(C) times wider.
        bandwidth = (model.inertia_ / rows.numel()) ** 0.5
        return KMeansConceptAutoencoder(
            centroids=centroids, bandwidth=bandwidth, batch_size=self.batch_size
        )
