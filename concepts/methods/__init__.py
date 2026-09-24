from typing import Protocol

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


class ConceptMethod(Protocol):
    def fit(self, z: LatentBatch, num_concepts: int) -> ConceptAutoencoder: ...
