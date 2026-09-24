from dataclasses import dataclass

import torch
from torch import Tensor, nn

from concepts.decomposition import TorchvisionDecomposition, _SpatialTail
from concepts.methods import ConceptAutoencoder
from concepts.typing import ConceptBatch, LatentBatch


@dataclass(frozen=True)
class LipschitzEstimate:
    """Empirical Lipschitz constant from the max growth ratio over sampled pairs.

    `point` is the largest `||out_i - out_j|| / ||in_i - in_j||` seen across all
    sampled pairs: a lower bound on the true constant that tightens as more pairs
    are sampled, never an upper bound or the exact value. `ci_low`/`ci_high` are a
    percentile interval over `num_trials` independent per-trial maxima, so they
    describe how much this *estimate* would move under a fresh sample of the same
    size -- not a confidence interval on the true (population) Lipschitz constant.
    """

    point: float
    ci_low: float
    ci_high: float
    num_pairs: int


def _ratios(
    in_a: Tensor, out_a: Tensor, in_b: Tensor, out_b: Tensor, eps: float = 1e-8
) -> Tensor:
    in_dist = (in_a - in_b).flatten(1).norm(dim=1)
    out_dist = (out_a - out_b).flatten(1).norm(dim=1)
    return out_dist[in_dist > eps] / in_dist[in_dist > eps]


def _summarize(
    trial_maxima: list[Tensor], all_ratios: list[Tensor], ci: float
) -> LipschitzEstimate:
    if not all_ratios:
        raise ValueError(
            "no valid pairs sampled (all pairs had near-zero input distance); "
            "increase pairs_per_trial, perturbation_scale, or check for "
            "duplicate/degenerate inputs"
        )
    all_ratios_t = torch.cat(all_ratios)
    trial_maxima_t = torch.stack(trial_maxima)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return LipschitzEstimate(
        point=all_ratios_t.max().item(),
        ci_low=torch.quantile(trial_maxima_t, lo_q).item(),
        ci_high=torch.quantile(trial_maxima_t, hi_q).item(),
        num_pairs=all_ratios_t.numel(),
    )


def estimate_lipschitz(
    inputs: Tensor,
    outputs: Tensor,
    num_trials: int = 20,
    pairs_per_trial: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> LipschitzEstimate:
    """Sample-based Lipschitz estimate of `inputs -> outputs` (batch dim 0 in both).

    Pairs are drawn across distinct rows of the same batch, so this is only
    valid when `outputs` is a single fixed function of `inputs` shared by every
    row (e.g. `g` applied to a batch of latents). It does not require a
    differentiable map. See `estimate_g_lipschitz` (uses this) and
    `estimate_gamma_curvature` (does not -- see its docstring for why).
    """
    n = inputs.shape[0]
    generator = torch.Generator().manual_seed(seed)
    trial_maxima = []
    all_ratios = []
    for _ in range(num_trials):
        idx_a = torch.randint(0, n, (pairs_per_trial,), generator=generator).to(
            inputs.device
        )
        offset = torch.randint(1, n, (pairs_per_trial,), generator=generator).to(
            inputs.device
        )
        idx_b = (idx_a + offset) % n  # offset in [1,n) guarantees idx_a != idx_b
        ratios = _ratios(inputs[idx_a], outputs[idx_a], inputs[idx_b], outputs[idx_b])
        if ratios.numel() == 0:
            continue
        trial_maxima.append(ratios.max())
        all_ratios.append(ratios)
    return _summarize(trial_maxima, all_ratios, ci)


def estimate_g_lipschitz(
    decomposition: TorchvisionDecomposition, z: LatentBatch, **kwargs: float | int
) -> LipschitzEstimate:
    """Estimates L_g from Theorem 1: the Lipschitz constant of the prediction head g.

    `g` is one fixed function of `z` regardless of any downstream class choice,
    so cross-sample pairs are valid here (unlike for M, see
    `estimate_gamma_curvature`).
    """
    with torch.no_grad():
        f_z = decomposition.predict_from_latent(z)
    return estimate_lipschitz(z, f_z, **kwargs)


def exact_g_lipschitz(
    decomposition: TorchvisionDecomposition, latent_shape: torch.Size
) -> float:
    """Exact L_g (Theorem 1) for boundaries where g is affine.

    - penultimate: g(z) = Wz + b, so L_g = ||W||_2 (largest singular value).
    - resnet layer4: g(z) = W avgpool(z) + b with z in R^{C x S}. The pooling map
      A = (1/S)[I ... I] has A A^T = I/S, so ||W A||_2 = ||W||_2 / sqrt(S).

    Any other g (remaining conv blocks at layer1-3) raises, since its Lipschitz
    constant has no closed form.
    """
    g = decomposition.g
    if isinstance(g, nn.Linear):
        head, num_positions = g, 1
    elif (
        isinstance(g, _SpatialTail)
        and len(g.blocks) == 0
        and isinstance(g.pool, nn.AdaptiveAvgPool2d)
        and isinstance(g.head, nn.Linear)
    ):
        head, num_positions = g.head, int(torch.Size(latent_shape[2:]).numel())
    else:
        raise NotImplementedError(
            f"exact L_g needs an affine g; boundary '{decomposition.boundary}' is not"
        )
    spectral_norm = torch.linalg.matrix_norm(head.weight.detach().double(), ord=2)
    return spectral_norm.item() / num_positions**0.5


def _grad_at(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    v: Tensor,
    target_index: Tensor,
) -> Tensor:
    """Per-row gradient of gamma_i = P_{target_index[i]} o g o D at v[i].

    Rows are independent, so this runs one chunk of samples at a time on the
    decomposition's device (one small autograd graph per chunk, instead of a
    graph through D(v) for every row at once) and returns on `v`'s device.
    """
    device = decomposition.device
    grads = []
    for v_chunk, target_chunk in zip(
        v.split(decomposition.batch_size),
        target_index.split(decomposition.batch_size),
        strict=True,
    ):
        v_chunk = v_chunk.detach().to(device).requires_grad_(True)
        out = decomposition.g(autoencoder.decode(v_chunk))
        rows = torch.arange(v_chunk.shape[0], device=device)
        scalar_out = out[rows, target_chunk.to(device)]
        (grad_v,) = torch.autograd.grad(scalar_out.sum(), v_chunk)
        grads.append(grad_v.detach().to(v.device))
    return torch.cat(grads)


def estimate_gamma_curvature(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    u: ConceptBatch,
    target_index: Tensor,
    num_trials: int = 20,
    pairs_per_trial: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    perturbation_scale: float = 1e-2,
) -> LipschitzEstimate:
    """Estimates M from Theorem 2: the gradient-Lipschitz constant of gamma=g o D.

    `target_index[i]` (each sample's own explained class, matching
    `insertion_total_attribution`) generally differs across samples, so gamma is a
    *different* scalar function per row: gamma_i = P_{target_index[i]} o g o D.
    Cross-sample pairs (as in `estimate_lipschitz`) would then measure how much
    gradients differ ACROSS classes, not the curvature of any single gamma_i --
    e.g. an affine decoder gives exactly zero true curvature, but cross-sample
    pairing reports a large, meaningless "M" purely from comparing different
    classes' (constant) gradients.

    Instead, this perturbs each row locally -- v_i vs v_i + delta_i, with
    target_index[i] held fixed -- so every pair compares the same gamma_i at
    two nearby points, which is what M actually bounds. `delta_i` has a random
    direction and norm `perturbation_scale * (||v_i|| + eps)`, i.e. a step
    proportional to that row's own representation scale, since concept
    representations from different methods can have very different magnitudes.
    """
    n = u.shape[0]
    generator = torch.Generator().manual_seed(seed)
    grad_base = _grad_at(decomposition, autoencoder, u, target_index)

    trial_maxima = []
    all_ratios = []
    for _ in range(num_trials):
        idx = torch.randint(0, n, (pairs_per_trial,), generator=generator).to(u.device)
        u_idx = u[idx]
        u_idx_flat = u_idx.flatten(1)  # matches u's real shape, spatial or not
        direction = torch.randn(u_idx_flat.shape, generator=generator).to(u.device)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        step = perturbation_scale * (u_idx_flat.norm(dim=1, keepdim=True) + 1e-6)
        u_perturbed = u_idx + (step * direction).view_as(u_idx)
        grad_perturbed = _grad_at(
            decomposition, autoencoder, u_perturbed, target_index[idx]
        )
        ratios = _ratios(u_idx, grad_base[idx], u_perturbed, grad_perturbed)
        if ratios.numel() == 0:
            continue
        trial_maxima.append(ratios.max())
        all_ratios.append(ratios)
    return _summarize(trial_maxima, all_ratios, ci)
