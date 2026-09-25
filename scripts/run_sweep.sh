#!/usr/bin/env bash
# Paper grid: 2 ResNet-50 boundaries x 5 concept-discovery methods x 3 concept counts x 3 seeds,
# 25,000 ImageNet-1k validation images per seed. Each combination is one Hydra job that logs
# its metrics to MLflow (see configs/concepts.yaml metadata.hyperparams for the params logged).
#
# Usage: scripts/run_sweep.sh [extra hydra overrides...]
# Example: scripts/run_sweep.sh data=imagenette data.num_samples=300   # cheap smoke test

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python -m concepts.bounds -m \
  model=resnet50 \
  model.decomposition.boundary=layer4,penultimate \
  method=pca,nmf,kmeans,sae,nonlinear_ae \
  eval.num_concepts=5,25,50 \
  eval.seed=0,1,2 \
  data=imagenet \
  data.num_samples=25000 \
  "$@"
