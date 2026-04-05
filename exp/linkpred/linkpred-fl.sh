#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/../../ || exit

EXP_SETTING='link-fl'
TASK='link-pred'
TRAINING_TYPE='forward'
FORWARD_TYPE='FL'

NUM_RUNS=5
SEED=100
EPOCHS=1000
VAL_EVERY=2
PATIENCE=100
NUM_HIDDEN=128
LR=0.001

for dataset in "CitationFull-CiteSeer" "CitationFull-Cora_ML" "CitationFull-PubMed" "Amazon-Photo" "GitHub"; do
  DATASET="${dataset}"

for model in "GCN" "SAGE" "GAT"; do

for num_layers in 4; do

python experiment.py \
--training-type "${TRAINING_TYPE}" \
--forward-type "${FORWARD_TYPE}" \
--exp-setting "${EXP_SETTING}" \
--dataset "${DATASET}" \
--task "${TASK}" \
--model "${model}" \
--num-layers "${num_layers}" \
--num-runs "${NUM_RUNS}" --seed "${SEED}" --epochs "${EPOCHS}" --val-every "${VAL_EVERY}" \
--lr "${LR}" --patience "${PATIENCE}" --num-hidden "${NUM_HIDDEN}" \
--overwrite-result "$@"

done

done

done