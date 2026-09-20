from dataclasses import dataclass

import torch
from torch import Tensor

from concepts.decomposition import TorchvisionDecomposition
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
        idx_a = torch.randint(0, n, (pairs_per_trial,), generator=generator)
        offset = torch.randint(1, n, (pairs_per_trial,), generator=generator)
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


def _grad_at(
    decomposition: TorchvisionDecomposition,
    autoencoder: ConceptAutoencoder,
    v: Tensor,
    target_index: Tensor,
) -> Tensor:
    v = v.detach().clone().requires_grad_(True)
    out = decomposition.predict_from_latent(autoencoder.decode(v))
    rows = torch.arange(v.shape[0])
    scalar_out = out[rows, target_index]
    (grad_v,) = torch.autograd.grad(scalar_out.sum(), v)
    return grad_v.detach()


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
    `attribution_error`) generally differs across samples, so gamma is a
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
        idx = torch.randint(0, n, (pairs_per_trial,), generator=generator)
        direction = torch.randn((pairs_per_trial, u.shape[1]), generator=generator)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        step = perturbation_scale * (u[idx].norm(dim=1, keepdim=True) + 1e-6)
        u_perturbed = u[idx] + step * direction
        grad_perturbed = _grad_at(
            decomposition, autoencoder, u_perturbed, target_index[idx]
        )
        ratios = _ratios(u[idx], grad_base[idx], u_perturbed, grad_perturbed)
        if ratios.numel() == 0:
            continue
        trial_maxima.append(ratios.max())
        all_ratios.append(ratios)
    return _summarize(trial_maxima, all_ratios, ci)
