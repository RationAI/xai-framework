from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torchvision.models import get_model, get_model_weights
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from concepts.typing import LatentBatch, OutputBatch


_RESNET_STAGES = ["layer1", "layer2", "layer3", "layer4"]


class _SpatialTail(nn.Module):
    """The real remaining top-level blocks after a stage boundary: g(z) for spatial z."""

    def __init__(
        self, blocks: list[nn.Module], pool: nn.Module, head: nn.Module
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.pool = pool
        self.head = head

    def forward(self, z: Tensor) -> Tensor:
        for block in self.blocks:
            z = block(z)
        return self.head(torch.flatten(self.pool(z), 1))


def _resnet_boundary(backbone: nn.Module, boundary: str) -> tuple[str, nn.Module]:
    fc = cast("nn.Module", backbone.fc)
    if boundary == "penultimate":
        return "flatten", fc
    if boundary not in _RESNET_STAGES:
        raise ValueError(
            f"resnet boundary must be one of {[*_RESNET_STAGES, 'penultimate']}, got '{boundary}'"
        )

    _, eval_nodes = get_graph_node_names(backbone)
    feature_node = next(n for n in reversed(eval_nodes) if n.startswith(boundary + "."))
    remaining = _RESNET_STAGES[_RESNET_STAGES.index(boundary) + 1 :]
    blocks = [cast("nn.Module", getattr(backbone, stage)) for stage in remaining]
    avgpool = cast("nn.Module", backbone.avgpool)
    return feature_node, _SpatialTail(blocks, avgpool, fc)


def _vgg_boundary(backbone: nn.Module, boundary: str) -> tuple[str, nn.Module]:
    if boundary != "penultimate":
        raise NotImplementedError(
            "vgg only supports the 'penultimate' boundary for now"
        )
    classifier = cast("nn.Sequential", backbone.classifier)
    return "classifier.5", classifier[6]


_BOUNDARY_BUILDERS: dict[str, Callable[[nn.Module, str], tuple[str, nn.Module]]] = {
    "resnet50": _resnet_boundary,
    "vgg16": _vgg_boundary,
}


class TorchvisionDecomposition(nn.Module):
    """f = g o h split at a named `boundary` ('penultimate', or for resnet also layer1-4).

    Earlier boundaries make h(x) spatial ([N,C,H,W]); g always stays the real
    remaining blocks, since resnet/vgg have no cross-stage skip connections.
    """

    def __init__(
        self,
        name: str,
        architecture: str,
        weights: str,
        boundary: str = "penultimate",
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        if architecture not in _BOUNDARY_BUILDERS:
            raise ValueError(
                f"no boundary builder registered for architecture '{architecture}'"
            )

        self.name = name
        self.boundary = boundary
        self.batch_size = batch_size

        weights_enum = get_model_weights(architecture)[weights]
        backbone = get_model(architecture, weights=weights_enum).eval()
        for param in backbone.parameters():
            param.requires_grad_(False)

        feature_node, tail = _BOUNDARY_BUILDERS[architecture](backbone, boundary)
        self.feature_node = feature_node
        self.h = create_feature_extractor(backbone, return_nodes={feature_node: "z"})
        self.g = tail

    def extract_latents(self, x: Tensor) -> LatentBatch:
        return self.h(x)["z"]

    def predict_from_latent(self, z: LatentBatch) -> OutputBatch:
        """Runs g in chunks of `batch_size` samples (conv tail activations get huge
        otherwise, e.g. for spatial layer1-4 boundaries on a whole eval split at
        once). Uses torch.split/cat, so this stays autograd-compatible for callers
        that need gradients through g (see estimate_gamma_curvature)."""
        if z.shape[0] <= self.batch_size:
            return self.g(z)
        return torch.cat([self.g(chunk) for chunk in z.split(self.batch_size)], dim=0)

    def forward(self, x: Tensor) -> OutputBatch:
        return self.predict_from_latent(self.extract_latents(x))
