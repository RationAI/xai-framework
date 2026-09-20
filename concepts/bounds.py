import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch import manual_seed

from concepts import metrics
from concepts.caching import extract_and_cache_latents


@hydra.main(config_path="../configs", config_name="concepts", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    manual_seed(config.eval.seed)
    logger.log_hyperparams(OmegaConf.to_container(config.metadata.hyperparams, resolve=True))

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    decomposition = hydra.utils.instantiate(config.model.decomposition).to(device)
    loader = hydra.utils.instantiate(config.data.loader)

    z, f_x, _labels = extract_and_cache_latents(
        decomposition,
        loader,
        cache_dir=f"{config.project_path}/cache",
        data_name=config.data.name,
        num_samples=config.data.num_samples,
        device=device,
    )
    # cache is always stored on cpu (see extract_and_cache_latents); move onto
    # the run's device here so it applies on both a fresh extraction and a cache hit
    z, f_x = z.to(device), f_x.to(device)

    num_fit = int(config.eval.fit_fraction * z.shape[0])
    z_fit, z_eval = z[:num_fit], z[num_fit:]
    f_fit, f_eval = f_x[:num_fit], f_x[num_fit:]

    method = hydra.utils.instantiate(config.method.extractor)
    autoencoder = method.fit(z_fit, config.eval.num_concepts)

    u_fit = autoencoder.encode(z_fit)
    u_eval = autoencoder.encode(z_eval)
    z_hat_eval = autoencoder.decode(u_eval)
    f_a_eval = decomposition.predict_from_latent(z_hat_eval)

    re = metrics.reconstruction_error(z_eval, z_hat_eval)
    fe = metrics.fidelity_error(f_eval, f_a_eval)

    probe = hydra.utils.instantiate(config.eval.probe)
    probe.fit(u_fit, f_fit)
    mce = metrics.model_completeness_error(probe(u_eval), f_eval)

    target_index = f_eval.argmax(dim=-1)
    pointwise_ae = metrics.attribution_error(
        decomposition, autoencoder, f_eval, u_eval, target_index
    )
    ae = metrics.mean_squared_attribution_error(pointwise_ae)

    lipschitz_kwargs = {
        "num_trials": config.eval.lipschitz.num_trials,
        "pairs_per_trial": config.eval.lipschitz.pairs_per_trial,
        "ci": config.eval.lipschitz.ci,
        "seed": config.eval.seed,
    }
    l_g = metrics.estimate_g_lipschitz(decomposition, z_eval, **lipschitz_kwargs)
    m = metrics.estimate_gamma_curvature(
        decomposition,
        autoencoder,
        u_eval,
        target_index,
        perturbation_scale=config.eval.lipschitz.perturbation_scale,
        **lipschitz_kwargs,
    )

    results = {
        "RE": re.item(),
        "FE": fe.item(),
        "MCE": mce.item(),
        "AE": ae.item(),
        # Theorem 1 guarantees MCE <= FE for any probe, since gamma=g o D is one
        # candidate in the infimum. A 0 here means the probe underfit relative
        # to that baseline, not that the bound itself failed.
        "MCE_leq_FE": float(mce.item() <= fe.item()),
        "L_g": l_g.point,
        "L_g_ci_low": l_g.ci_low,
        "L_g_ci_high": l_g.ci_high,
        "M": m.point,
        "M_ci_low": m.ci_low,
        "M_ci_high": m.ci_high,
        # L_g/M are sample-based lower bounds on the true constants (see
        # concepts/metrics/lipschitz.py), so a 0 here means the sampled L_g was
        # too small to certify the bound -- not that Theorem 1 itself failed.
        "FE_leq_Lg2_RE": float(fe.item() <= l_g.point**2 * re.item()),
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
