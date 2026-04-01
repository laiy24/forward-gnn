#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/../../ || exit

EXP_SETTING='node-sf-top2input'
TASK='node-class'
TRAINING_TYPE='forward'
FORWARD_TYPE='SF'
TOPDOWN_TYPE='top2input'

NUM_RUNS=5
SEED=100
EPOCHS=1000
NUM_HIDDEN=128
LR=0.001
APPEND_LABEL=none
VAL_EVERY=2
PATIENCE=100
declare -a DATASETS=("CitationFull-CiteSeer" "CitationFull-Cora_ML" "CitationFull-PubMed" "Amazon-Photo" "GitHub")

for model in "GNN-GCN" "GNN-SAGE" "GNN-GAT"; do

for dataset in "${DATASETS[@]}"; do
  DATASET="${dataset}"

for num_layers in 1 2 3 4; do

python experiment.py \
--training-type "${TRAINING_TYPE}" \
--forward-type "${FORWARD_TYPE}" \
--topdown-model "${TOPDOWN_TYPE}" \
--exp-setting "${EXP_SETTING}" \
--dataset "${DATASET}" \
--task "${TASK}" \
--model "${model}" \
--num-layers "${num_layers}" \
--num-runs "${NUM_RUNS}" --seed "${SEED}" --epochs "${EPOCHS}" --val-every "${VAL_EVERY}" \
--lr "${LR}" --patience "${PATIENCE}" --num-hidden "${NUM_HIDDEN}" \
--append-label "${APPEND_LABEL}" \
--alternating-update \
--aug-edge-direction bidirection \
--overwrite-result "$@"

done

done

done