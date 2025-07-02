#!/bin/bash
set -euxo pipefail

git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .

cd ..

git clone https://github.com/state-spaces/mamba.git
cd mamba
MAMBA_FORCE_BUILD=TRUE pip install .

echo "export PATH=${HOME}/.local/bin${PATH:+:${PATH}}" | sudo tee -a /etc/bash.bashrc
source /etc/bash.bashrc