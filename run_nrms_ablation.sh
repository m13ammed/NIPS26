#!/usr/bin/env bash
set -euo pipefail

# Runs cumulative nRMSE ablations for PDE LTD.
# You can pass extra CLI overrides to every run, e.g.:
# ./run_nrms_ablation.sh trainer.params.devices=1 trainer.batch_size=8

CONFIG="configs/pde-ltd-mc-mse.yaml"
BASE_NAME="pde-ltd-mc-xs-mse-nrmse-ablation"
PROJECT_OVERRIDE=("project=nRMSE Ablation")

STAGE_TAGS=(
  "output_act"
  "upsample_act"
  "l2_drop"
  "gated_mlp"
  "gated_fusion"
  "higher_lr"
  # "muon"
  "lr_scheduler"
  "warmup100"
)

STAGE_OVERRIDES=(
  "model.params.model.params.output_activation=gelu"
  "model.params.model.params.use_upsample_activation=True"
  "model.params.model.params.sprint_drop_mode=l2"
  "model.params.model.params.use_gated_mlp=True"
  "model.params.model.params.sprint_fusion_type=gated"
  "trainer.base_learning_rate=5.5e-4"
  # "model.params.optimizer=muon trainer.muon_learning_rate=1.0e-2"
  "model.params.lr_scheduler=linear"
  "model.params.warmup_steps=100"
)

EXTRA_OVERRIDES=("$@")
CUMULATIVE_OVERRIDES=()

for i in "${!STAGE_TAGS[@]}"; do
  stage_tag="${STAGE_TAGS[$i]}"

  read -r -a stage_items <<< "${STAGE_OVERRIDES[$i]}"
  CUMULATIVE_OVERRIDES+=("${stage_items[@]}")

  run_name="${BASE_NAME}-all_ex_$(printf "%02d" "$((i + 1))")-${stage_tag}-full-bal"

  echo
  echo "============================================================"
  echo "[$((i + 1))/${#STAGE_TAGS[@]}] Running stage: ${stage_tag}"
  echo "Run name: ${run_name}"
  echo "New overrides: ${STAGE_OVERRIDES[$i]}"
  echo "Cumulative overrides: ${CUMULATIVE_OVERRIDES[*]}"
  echo "============================================================"

  python main.py \
    -c "${CONFIG}" \
    -n "${run_name}" \
    "${PROJECT_OVERRIDE[@]}" \
    "${CUMULATIVE_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
done
