#!/usr/bin/env bash
# Full experiment grid: 5 ResNet50 boundaries x 5 concept-discovery methods x 3 concept counts.
# Each combination is one Hydra job that computes RE/FE/MCE/AE and logs them to MLflow
# (see configs/concepts.yaml metadata.hyperparams for the params logged per run).
#
# Usage: scripts/run_sweep.sh [extra hydra overrides...]
# Example: scripts/run_sweep.sh data=imagenette   # cheap smoke test before the real sweep

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python -m concepts.bounds -m \
  model=resnet50 \
  model.decomposition.boundary=layer1,layer2,layer3,layer4,penultimate \
  method=pca,nmf,kmeans,sae,nonlinear_ae \
  eval.num_concepts=5,25,50 \
  data=imagenet \
  "$@"
