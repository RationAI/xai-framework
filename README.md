# XAI Framework: Empirical Completeness Bounds

This project empirically tests the theoretical completeness bounds for concept-based
explainable AI (C-XAI) introduced in the accompanying paper: given a trained model
`f = g o h` and a `k`-concept autoencoder `A = (E, D)` fitted on its latent space, it
measures four quantities per (model, boundary, concept-extraction method, `k`) combination:

-   **RE** — reconstruction error of the concept autoencoder
-   **FE** — fidelity error between `f` and the induced concept-autoencoder model `f_A`
-   **MCE** — model completeness error (via a fitted probe on the concept representations)
-   **AE** — mean squared attribution error of per-concept contributions

Theorem 1 in the paper guarantees `MCE <= FE <= L_g^2 * RE`; this repo runs the pipeline
across models, boundaries, methods, and concept counts to check how tightly that bound
holds in practice.

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
├── eval                 # num_concepts (k), fit/eval split, seed, completeness probe
├── hydra
│   ├── default.yaml
│   └── job_logging/custom.yaml
├── logger/mlflow.yaml
├── base.yaml
└── concepts.yaml
```

For a resnet50/vgg16 decomposition, the `model.decomposition.boundary` override selects
where the model is split into `h`/`g` (`layer1`-`layer4` or `penultimate` for resnet50;
`penultimate` only for vgg16 so far).

Please do not delete the `configs/hydra` directory as it is required for Hydra execution.

## Usage

-   **Run a single configuration:**
    ```bash
    uv run python -m concepts.bounds model=resnet50 data=imagenette method=pca eval.num_concepts=5
    ```
-   **Run the full experiment grid** (5 boundaries x 5 methods x 3 concept counts on resnet50/imagenet):
    ```bash
    scripts/run_sweep.sh
    ```
    Each combination is a separate Hydra job (`-m` multirun) that logs RE/FE/MCE/AE and its
    hyperparameters to the MLflow experiment named in `configs/base.yaml`
    (`metadata.experiment_name`), and writes `metrics.json` to
    `{project_path}/cache/runs/{run_id}/`.
-   **Submit a job to the cluster** instead of running locally, see [`scripts/run_concepts.py`](scripts/run_concepts.py).
-   **Aggregate results** from MLflow into a comparison table and per-metric plots:
    ```bash
    uv run python scripts/aggregate_results.py
    ```

## Linting, Formatting and Type Checking:

```bash
uvx ruff check  # Check and fix linting issues
uvx ruff format # Format code
uv run mypy .     # Run mypy
```
