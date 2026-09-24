import torch
from sklearn.decomposition import NMF as SKNMF

from concepts.methods.common import apply_batched, from_rows, to_rows, zero_all_but
from concepts.typing import ConceptBatch, LatentBatch


class NMFConceptAutoencoder:
    """NMF concept autoencoder: fixed nonnegative basis, per-input coefficients via NNLS.

    Matches the framework's ICE example (learn U on a fitting set, hold it
    fixed, solve for each new input's coefficients).
    """

    def __init__(
        self, model: SKNMF, components: LatentBatch, batch_size: int = 4096
    ) -> None:
        self._model = model  # fitted sklearn NMF, used to project new latents
        self.components = (
            components  # [k, C] nonneg rows, torch view of model.components_
        )
        self.num_concepts = components.shape[0]
        self.batch_size = batch_size
        self.affine_decoder = True

    def encode(self, z: LatentBatch) -> ConceptBatch:
        rows = to_rows(z).clamp(min=0)  # NMF requires nonnegative input

        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            return torch.from_numpy(
                self._model.transform(chunk.detach().cpu().numpy())
            ).to(chunk)

        u = apply_batched(_transform, rows, self.batch_size)
        return from_rows(u, z)

    def decode(self, u: ConceptBatch) -> LatentBatch:
        rows = to_rows(u).clamp(min=0)
        z = apply_batched(lambda r: r @ self.components, rows, self.batch_size)
        return from_rows(z, u)

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch:
        return zero_all_but(u, index)


class NMFMethod:
    def __init__(self, max_iter: int = 200, seed: int = 0, batch_size: int = 4096) -> None:
        self.max_iter = max_iter
        self.seed = seed
        self.batch_size = batch_size

    def fit(self, z: LatentBatch, num_concepts: int) -> NMFConceptAutoencoder:
        rows = to_rows(z).clamp(min=0)
        model = SKNMF(
            n_components=num_concepts,
            init="nndsvda",
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        model.fit(rows.detach().cpu().numpy())
        components = torch.from_numpy(model.components_).to(rows)
        return NMFConceptAutoencoder(
            model=model, components=components, batch_size=self.batch_size
        )
