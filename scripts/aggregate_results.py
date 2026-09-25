"""Collects concept-bounds runs from MLflow into one comparison table and figure set.

Usage:
    uv run python scripts/aggregate_results.py [--experiment "XAI Framework"] [--out-dir DIR]

Requires MLFLOW_TRACKING_URI (and credentials) to already be set in the
environment, same as when the runs were produced by `concepts.bounds`.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


_BOUNDARY_ORDER = ["layer1", "layer2", "layer3", "layer4", "penultimate"]
_METHOD_ORDER = ["pca", "nmf", "kmeans", "sae", "nonlinear_ae"]
_METHOD_COLOR = {
    "pca": "#2a78d6",  # blue
    "nmf": "#eb6834",  # orange
    "kmeans": "#1baf7a",  # aqua
    "sae": "#eda100",  # yellow
    "nonlinear_ae": "#e87ba4",  # magenta
}
_MCE_UB = ["MCE_UB_mlp", "MCE_UB_residual", "MCE_UB_gD"]
_RULES = ["insertion", "occlusion", "gradxinput"]
_ERRORS = [
    "RE",
    "RE_pooled",
    "FE",
    *_MCE_UB,
    "FE_target",
    *[f"ATE_{rule}" for rule in _RULES],
    *[f"ADD_{rule}" for rule in _RULES],
]
_BOUNDS = ["FE_bound", "ADD_bound", "ATE_bound"]
_METRICS = [f"{name}_RMSE" for name in _ERRORS + _BOUNDS] + ["L_g", "M"]


def fetch_results(experiment_name: str) -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"no MLflow experiment named '{experiment_name}' found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], output_format="pandas"
    )
    if runs.empty:
        raise ValueError(f"experiment '{experiment_name}' has no runs")

    keep = {
        "params.model": "model",
        "params.boundary": "boundary",
        "params.method": "method",
        "params.num_concepts": "num_concepts",
        "params.data": "data",
        "params.num_samples": "num_samples",
        "params.seed": "seed",
        **{
            f"metrics.{name}_{scale}": f"{name}_{scale}"
            for name in _ERRORS + _BOUNDS
            for scale in ("MSE", "RMSE")
        },
        **{
            f"metrics.{name}": name
            for name in [
                "L_g",
                "L_g_sampled",
                "L_g_sampled_ci_low",
                "L_g_sampled_ci_high",
                "M",
                "M_ci_low",
                "M_ci_high",
                "C4_root",
                "rho_FE",
                "alpha",
                "kappa",
                "FE_leq_bound",
                *[f"{name}_leq_FE" for name in _MCE_UB],
                *[f"ADD_{rule}_leq_bound" for rule in _RULES],
                *[f"ATE_{rule}_leq_bound" for rule in _RULES],
            ]
        },
    }
    missing = [c for c in keep if c not in runs.columns]
    if missing:
        raise ValueError(
            f"runs are missing expected columns {missing}; were they logged with "
            "the metadata.hyperparams block in configs/concepts.yaml?"
        )

    df = runs[list(keep)].rename(columns=keep).dropna(subset=["boundary", "method"])
    df["num_concepts"] = df["num_concepts"].astype(int)
    df["boundary"] = pd.Categorical(
        df["boundary"], categories=_BOUNDARY_ORDER, ordered=True
    )
    df["method"] = pd.Categorical(df["method"], categories=_METHOD_ORDER, ordered=True)
    return df.sort_values(["boundary", "method", "num_concepts"]).reset_index(drop=True)


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    k_values = sorted(df["num_concepts"].unique())
    fig, axes = plt.subplots(
        1, len(k_values), figsize=(4.5 * len(k_values), 3.5), sharey=True
    )
    axes = [axes] if len(k_values) == 1 else axes

    boundaries = [b for b in _BOUNDARY_ORDER if b in df["boundary"].unique()]
    methods = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    x = range(len(boundaries))
    width = 0.8 / max(len(methods), 1)

    for ax, k in zip(axes, k_values, strict=True):
        subset = df[df["num_concepts"] == k]
        for i, method in enumerate(methods):
            values = [
                subset.loc[
                    (subset["boundary"] == b) & (subset["method"] == method), metric
                ].mean()
                for b in boundaries
            ]
            offsets = [xi + (i - (len(methods) - 1) / 2) * width for xi in x]
            ax.bar(
                offsets, values, width=width, label=method, color=_METHOD_COLOR[method]
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(boundaries, rotation=30, ha="right")
        ax.set_title(f"k={k}")
        ax.set_yscale("log")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(f"{metric} (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)
    fig.suptitle(f"{metric} across boundary x method x k", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="XAI Framework")
    parser.add_argument(
        "--out-dir", default="/mnt/projects/xai_framework/cache/aggregated"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_results(args.experiment)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {len(df)} rows to {csv_path}")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for metric in _METRICS:
        plot_path = plots_dir / f"{metric.lower()}.png"
        plot_metric(df, metric, plot_path)
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
