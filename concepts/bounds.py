import gc
import json
import traceback
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch import manual_seed

from concepts import metrics
from concepts.caching import extract_and_cache_latents
from concepts.methods.common import apply_batched
from concepts.probes.decoded_head import AffineDecodedHead


@hydra.main(config_path="../configs", config_name="concepts", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    try:
        _run(config, logger)
    except Exception as exc:
        # Hydra multirun runs every job sequentially in one process; MLflow's
        # fluent API only allows one active run per process, so the next job's
        # mlflow.start_run() would fail with "already active" unless this run
        # is explicitly ended here (a single non-multirun job gets away without
        # this because the process exits right after, and mlflow's atexit hook
        # closes the dangling run silently).
        mlflow.end_run(status="FAILED")
        # The traceback keeps _run's frame, and with it every tensor it held,
        # alive while Hydra records the failure. Dropping those locals lets the
        # memory be freed below, so one OOM does not also fail every later
        # job of the multirun (seen on a 16GB P100).
        traceback.clear_frames(exc.__traceback__)
        raise
    else:
        mlflow.end_run(status="FINISHED")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run(config: DictConfig, logger: MLFlowLogger) -> None:
    manual_seed(config.eval.seed)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = OmegaConf.to_container(config.metadata.hyperparams, resolve=True)
    hyperparams["device"] = device
    if device == "cuda":
        hyperparams["gpu_name"] = torch.cuda.get_device_name(0)
    logger.log_hyperparams(hyperparams)

    decomposition = hydra.utils.instantiate(config.model.decomposition).to(device)
    loader = hydra.utils.instantiate(config.data.loader)

    z, f_x, _labels = extract_and_cache_latents(
        decomposition,
        loader,
        cache_dir=f"{config.project_path}/cache",
        data_name=config.data.name,
        num_samples=config.data.num_samples,
        seed=config.data.loader.seed,
        device=device,
    )
    # z stays on cpu (the cache always is, see extract_and_cache_latents): at
    # layer4 it is ~10GB for 25k samples, more than many gpus hold. Everything
    # latent-sized below streams chunks of samples to `device` instead. The
    # outputs f(x), and later the concept representations u, are small enough
    # to live on `device` whole.
    f_x = f_x.to(device)

    num_fit = int(config.eval.fit_fraction * z.shape[0])
    z_fit, z_eval = z[:num_fit], z[num_fit:]
    f_fit, f_eval = f_x[:num_fit], f_x[num_fit:]

    # Reseed: a fresh latent extraction (cache miss) consumes global RNG draws in
    # the DataLoader that a cache hit does not, which would otherwise make
    # randomized fits (e.g. torch.pca_lowrank) depend on the cache state.
    manual_seed(config.eval.seed)
    method = hydra.utils.instantiate(config.method.extractor)
    autoencoder = method.fit(z_fit, config.eval.num_concepts, device=device)

    # no_grad: encode/decode/g build a backward graph whenever the method has
    # learnable weights (SAE/NonlinearAE), even though nothing here calls
    # .backward() -- that's pure wasted activation memory.
    batch_size = decomposition.batch_size
    with torch.no_grad():
        u_fit = apply_batched(autoencoder.encode, z_fit, batch_size, device).to(device)
        u_eval = apply_batched(autoencoder.encode, z_eval, batch_size, device).to(
            device
        )
        f_a_eval = metrics.decoded_head(decomposition, autoencoder, u_eval)
        f_a_fit = metrics.decoded_head(decomposition, autoencoder, u_fit)
        re = metrics.reconstruction_error(
            autoencoder, z_eval, u_eval, batch_size, device
        )
        re_pooled = metrics.pooled_reconstruction_error(
            autoencoder, z_eval, u_eval, batch_size, device
        )
    fe = metrics.fidelity_error(f_eval, f_a_eval)

    # MCE is an infimum over all heads gamma; each fitted head only gives an
    # upper bound on it, so these are reported as MCE_UB_*, never as MCE itself.
    #   mlp:      MLP on u from random init.
    #   residual: g o D + an MLP correction starting at zero.
    #   gD:       g' o D' refit from g o D -- closed-form least squares when D is
    #             affine (the family is then exactly the affine maps of pooled u),
    #             gradient fine-tuning when D is nonlinear.
    probes = config.eval.probes
    mlp = hydra.utils.instantiate(probes.mlp).fit(u_fit, f_fit)
    residual = hydra.utils.instantiate(probes.residual).fit(u_fit, f_a_fit, f_fit)
    if autoencoder.affine_decoder:
        gd_head = AffineDecodedHead().fit(u_fit, f_fit)
    else:
        gd_head = hydra.utils.instantiate(probes.finetuned_gD).fit(
            u_fit, f_fit, autoencoder.decoder, decomposition.g
        )
    mce_ub = {
        "mlp": metrics.model_completeness_error(mlp(u_eval), f_eval),
        "residual": metrics.model_completeness_error(
            residual(u_eval, f_a_eval), f_eval
        ),
        "gD": metrics.model_completeness_error(gd_head(u_eval), f_eval),
    }

    # Attribution completeness is stated for a scalar output (N=1): each sample
    # explains its predicted class, so FE/ATE/ADD below are all at target_index.
    target_index = f_eval.argmax(dim=-1)
    rows = torch.arange(f_eval.shape[0], device=f_eval.device)
    f_target, f_a_target = f_eval[rows, target_index], f_a_eval[rows, target_index]
    total_attributions = metrics.total_attributions(
        decomposition, autoencoder, u_eval, target_index
    )
    fe_target = torch.mean((f_target - f_a_target) ** 2)
    ate = {
        rule: torch.mean((f_target - total) ** 2)
        for rule, total in total_attributions.items()
    }
    add = {
        rule: torch.mean((f_a_target - total) ** 2)
        for rule, total in total_attributions.items()
    }

    lipschitz_kwargs = {
        "num_trials": config.eval.lipschitz.num_trials,
        "pairs_per_trial": config.eval.lipschitz.pairs_per_trial,
        "ci": config.eval.lipschitz.ci,
        "seed": config.eval.seed,
    }
    l_g = metrics.exact_g_lipschitz(decomposition, z_eval.shape)
    # Sampled L_g is a lower bound on the exact one; kept only as a sanity check.
    l_g_sampled = metrics.estimate_g_lipschitz(
        decomposition, z_eval, **lipschitz_kwargs
    )
    m = metrics.estimate_gamma_curvature(
        decomposition,
        autoencoder,
        u_eval,
        target_index,
        perturbation_scale=config.eval.lipschitz.perturbation_scale,
        **lipschitz_kwargs,
    )
    c4_root = metrics.representation_fourth_moment_root(u_eval).item()

    # Errors in the paper are RMSEs; the MSE (squared) versions are logged too.
    errors_mse = {
        "RE": re.item(),
        "FE": fe.item(),
        **{f"MCE_UB_{name}": value.item() for name, value in mce_ub.items()},
        "FE_target": fe_target.item(),
        **{f"ATE_{rule}": value.item() for rule, value in ate.items()},
        **{f"ADD_{rule}": value.item() for rule, value in add.items()},
        "RE_pooled": re_pooled.item(),
    }
    errors_rmse = {name: value**0.5 for name, value in errors_mse.items()}
    # Right-hand sides of the bounds, in RMSE form:
    #   FE <= L_g RE (Theorem 1), ADD <= M sqrt(E||C||^4) (Theorem 2),
    #   ATE <= FE + ADD bound (Corollary, with the scalar FE at target_index).
    # The ADD bound (and so the ATE bound) is the same for all three rules.
    # M is sample-based (a lower bound on the true constant), so the two
    # attribution bounds are estimates; the FE bound uses the exact L_g.
    bounds_rmse = {
        "FE_bound": l_g * errors_rmse["RE"],
        "ADD_bound": m.point * c4_root,
    }
    bounds_rmse["ATE_bound"] = errors_rmse["FE_target"] + bounds_rmse["ADD_bound"]

    # rho_FE = FE / (L_g RE) = alpha * kappa exactly (eq. cg-fidelity-slack),
    # with ||W||_2 = L_g sqrt(S) and S the number of spatial locations:
    #   alpha = FE / (||W||_2 RMSE(e_bar)),  kappa = sqrt(S) RMSE(e_bar) / RE.
    num_positions = z_eval[0, 0].numel() if z_eval.dim() > 2 else 1
    rho_fe = errors_rmse["FE"] / bounds_rmse["FE_bound"]
    alpha = errors_rmse["FE"] / (l_g * num_positions**0.5 * errors_rmse["RE_pooled"])
    kappa = num_positions**0.5 * errors_rmse["RE_pooled"] / errors_rmse["RE"]

    # float32 slack for the checks below: e.g. an affine gamma gives M = 0 and
    # ADD = 0 in exact arithmetic, but ADD comes out ~1e-7 numerically.
    tolerance = 1e-5 * f_target.pow(2).mean().sqrt().item()

    def leq(lhs: float, rhs: float) -> float:
        return float(lhs <= rhs + tolerance)

    results = {
        **{f"{name}_MSE": value for name, value in errors_mse.items()},
        **{f"{name}_RMSE": value for name, value in errors_rmse.items()},
        **{f"{name}_MSE": value**2 for name, value in bounds_rmse.items()},
        **{f"{name}_RMSE": value for name, value in bounds_rmse.items()},
        "L_g": l_g,
        "L_g_sampled": l_g_sampled.point,
        "L_g_sampled_ci_low": l_g_sampled.ci_low,
        "L_g_sampled_ci_high": l_g_sampled.ci_high,
        "M": m.point,
        "M_ci_low": m.ci_low,
        "M_ci_high": m.ci_high,
        "C4_root": c4_root,
        "rho_FE": rho_fe,
        "alpha": alpha,
        "kappa": kappa,
        "FE_leq_bound": leq(errors_rmse["FE"], bounds_rmse["FE_bound"]),
        # MCE <= FE holds by Proposition (g o D is one admissible head); a 0 here
        # only means that head did worse than g o D, not that the bound failed.
        **{
            f"MCE_UB_{name}_leq_FE": leq(
                errors_rmse[f"MCE_UB_{name}"], errors_rmse["FE"]
            )
            for name in mce_ub
        },
        **{
            f"ADD_{rule}_leq_bound": leq(
                errors_rmse[f"ADD_{rule}"], bounds_rmse["ADD_bound"]
            )
            for rule in metrics.ATTRIBUTION_RULES
        },
        **{
            f"ATE_{rule}_leq_bound": leq(
                errors_rmse[f"ATE_{rule}"], bounds_rmse["ATE_bound"]
            )
            for rule in metrics.ATTRIBUTION_RULES
        },
    }
    logger.log_metrics(results)

    artifact_dir = (
        Path(config.project_path) / "cache" / "runs" / (logger.run_id or "local")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.log_artifacts(str(artifact_dir), artifact_path="bounds")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
