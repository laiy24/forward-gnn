#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/ || exit

CONDA_ENV=ForwardLearningGNN

if conda info --envs | grep -q "${CONDA_ENV} "; then
  echo "\"${CONDA_ENV}\" conda env exists.";
else
  conda create -y --name "${CONDA_ENV}" python=3.11.5
fi

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}"/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

if [[ "${OSTYPE}" == "darwin"* ]]; then  # Mac OS
  conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
else
  conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.6 -c pytorch -c nvidia
fi

conda install -y pyg==2.6.1 -c pyg  # for Linux and OSX
conda install -y -c conda-forge tqdm