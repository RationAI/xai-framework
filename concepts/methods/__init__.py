from typing import Protocol, Self

import torch

from concepts.typing import ConceptBatch, LatentBatch


class ConceptAutoencoder(Protocol):
    """A=(E,D); concept i is coordinate i of the encoding, so `S_i` is a column mask."""

    num_concepts: int
    # True when D is affine on each row, so gamma = g o D is affine at the
    # penultimate/layer4 boundaries (see concepts/probes/decoded_head.py).
    affine_decoder: bool

    def encode(self, z: LatentBatch) -> ConceptBatch: ...

    def decode(self, u: ConceptBatch) -> LatentBatch: ...

    def zero_concept(self, u: ConceptBatch, index: int) -> ConceptBatch: ...

    def to(self, device: torch.device | str) -> Self: ...


class ConceptMethod(Protocol):
    def fit(
        self,
        z: LatentBatch,
        num_concepts: int,
        device: torch.device | str | None = None,
    ) -> ConceptAutoencoder:
        """Fits on `z` (which may stay on cpu); returns an autoencoder on `device`."""
        ...
