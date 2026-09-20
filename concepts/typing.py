from torch import Tensor


type LatentBatch = Tensor  # [N, d] flat latent vectors h(x), Z assumed subset of R^d
type OutputBatch = Tensor  # [N, out_dim] model outputs f(x)
type ConceptBatch = Tensor  # [N, k] concatenated scalar concept representations E(z)
