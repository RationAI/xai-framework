import torch
from sklearn.cluster import KMeans as SKKMeans

from concepts.methods.common import from_rows, to_rows, zero_all_but
from concepts.typing import ConceptBatch, LatentBatch


class KMeansConceptAutoencoder:
    """Concept i's activation is an RBF similarity to centroid i; decode is linear.

    Unlike PCA/NMF, this reconstruction isn't fit to minimize RE directly, so
    it's a useful contrast baseline rather than a strong one.
    """

    def __init__(self, centroids: LatentBatch, bandwidth: float) -> None:
        self.centroids = centroids  # [k, C]
        self.bandwidth = bandwidth
        self.num_concepts = centroids.shape[0]

    def encode(self, z: LatentBatch) -> ConceptBatch:
        rows = to_rows(z)
        dist_sq = torch.cdist(rows, self.centroids) ** 2
        u = torch.exp(-dist_sq / (2 * self.bandwidth**2))
        return from_rows(u, z)

    def decode(self, u: ConceptBatch) -> LatentBatch:
        rows = to_rows(u)
        z = rows @ self.centroids
        return from_rows(z, u)

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch:
        return zero_all_but(u, index)


class KMeansMethod:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, z: LatentBatch, num_concepts: int) -> KMeansConceptAutoencoder:
        rows = to_rows(z)
        model = SKKMeans(n_clusters=num_concepts, n_init="auto", random_state=self.seed)
        model.fit(rows.detach().cpu().numpy())
        centroids = torch.from_numpy(model.cluster_centers_).to(rows)
        bandwidth = (model.inertia_ / rows.shape[0]) ** 0.5
        return KMeansConceptAutoencoder(centroids=centroids, bandwidth=bandwidth)
