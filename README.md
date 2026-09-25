# XAI Framework: Empirical Tightness of Completeness Bounds

This project evaluates how tight the completeness bounds for concept-based explainable AI
(C-XAI) from the accompanying paper are in practice. Given a trained model `f = g o h`,
split at a boundary into `h` and the prediction head `g`, and a `k`-concept autoencoder
`A = (E, D)` fitted on the latents `h(x)`, the induced concept model is `f_A = g o D o E o h`.
All errors are root-mean-square errors (RMSE) on a held-out half of the data:

-   **RE**: reconstruction error of the concept autoencoder, `RMSE(h, D o E o h)`
-   **FE**: fidelity error, `RMSE(f, f_A)`
-   **MCE**: model completeness error, an infimum over all heads on the concept
    representations; only upper estimates `MCE_UB_*` from fitted heads are reported
-   **ATE / ADD**: attribution and additivity errors of the concept attributions
    (insertion, occlusion, gradient-times-input) with respect to `f` and `f_A`

The bounds under test are

-   `MCE <= FE <= L_g * RE`, with the exact Lipschitz constant `L_g` of an affine head
-   `ATE <= FE + ADD` and `ADD <= M * sqrt(E ||C(x)||^4)`, with a sampled estimate of the
    curvature constant `M`

For each bound, the pipeline logs both sides and whether the bound holds, together with
the fidelity tightness ratio `rho_FE = FE / (L_g * RE)` and its factors `alpha` and `kappa`.
Both RMSE and MSE are logged for every error.

The paper evaluates ResNet-50 (torchvision `IMAGENET1K_V2` weights) on 25,000 ImageNet-1k
validation images per seed, at the `layer4` and `penultimate` boundaries, where `g` is
affine and `L_g` has a closed form.

## Installation

1.  **Clone the repository** and `cd` into it.
2.  **Install uv:** follow the instructions on the [uv website](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it installed.
3.  **Install dependencies:**
    ```bash
    uv sync
    ```
    or if you need to submit a job to a cluster, use:
    ```bash
    uv sync --extra job
    ```
4.  **Configure MLflow:** set `MLFLOW_TRACKING_URI` and credentials (see `.env`).
5.  **Provide ImageNet-1k:** the loader reads the Hugging Face parquet release of
    ImageNet-1k (`data/validation-*.parquet` shards) from `{project_path}/data/imagenet-1k`,
    where `project_path` is set in [`configs/base.yaml`](configs/base.yaml).

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management, composed from
the [`configs/`](configs) directory. [`configs/concepts.yaml`](configs/concepts.yaml) is
the entrypoint config for the bounds pipeline; [`configs/base.yaml`](configs/base.yaml)
holds shared settings (MLflow experiment name, project storage path) that it inherits from.

Each run is defined by four composable axes, overridable from the command line:

```plaintext
configs
├── model                # Model + latent decomposition (f = g o h), e.g. resnet50, vgg16
├── data                 # Dataset + loader, e.g. imagenet, imagenette
├── method               # Concept extraction method, e.g. pca, nmf, kmeans, sae, nonlinear_ae
├── eval                 # num_concepts (k), fit/eval split, seed, MCE heads, Lipschitz/curvature estimation
├── hydra
│   ├── default.yaml
│   └── job_logging/custom.yaml
├── logger/mlflow.yaml
├── base.yaml
└── concepts.yaml
```

For a resnet50/vgg16 decomposition, the `model.decomposition.boundary` override selects
where the model is split into `h`/`g`. The bounds pipeline supports `layer4` and
`penultimate` for resnet50 and `penultimate` for vgg16, the boundaries at which `g` is
affine and its Lipschitz constant can be computed exactly.

The seed (`eval.seed`) determines the sampled images, the split into fitting and
evaluation halves, and the initialization of all methods and heads.

Please do not delete the `configs/hydra` directory as it is required for Hydra execution.

## Usage

-   **Run a single configuration:**
    ```bash
    uv run python -m concepts.bounds model=resnet50 model.decomposition.boundary=layer4 \
        data=imagenet data.num_samples=25000 method=pca eval.num_concepts=25 eval.seed=0
    ```
-   **Run the paper grid** (2 boundaries x 5 methods x 3 concept counts x 3 seeds on
    resnet50/imagenet, 90 runs):
    ```bash
    scripts/run_sweep.sh
    ```
    Each combination is a separate Hydra job (`-m` multirun) that logs its metrics and
    hyperparameters to the MLflow experiment named in `configs/base.yaml`
    (`metadata.experiment_name`) and writes `metrics.json` to
    `{project_path}/cache/runs/{run_id}/`. Latents are cached per model, boundary, dataset,
    sample count, and seed. The paper's runs used 64 GiB of RAM; the latents stay in
    RAM and are streamed to the GPU in chunks, so a 16 GB GPU is sufficient.
-   **Submit a job to the cluster** instead of running locally, see [`scripts/run_concepts.py`](scripts/run_concepts.py).
-   **Aggregate results** from MLflow into a per-run table (`results.csv`), a mean and
    standard deviation over seeds (`results_summary.csv`), and per-metric plots:
    ```bash
    uv run python scripts/aggregate_results.py --out-dir results/
    ```

## Linting, Formatting and Type Checking:

```bash
uvx ruff check  # Check and fix linting issues
uvx ruff format # Format code
uv run mypy .     # Run mypy
```
