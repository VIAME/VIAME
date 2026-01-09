#!/bin/bash

# Setup VIAME path (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup input paths (note: these must be absolute paths currently)

CURRENT_DIR=$(readlink -f $(dirname $BASH_SOURCE[0]))

export DATA_FOLDER=$CURRENT_DIR/training_data
export TRAIN_FOLDER=$CURRENT_DIR/deep_tracking
export GPU_COUNT=1
export THRESH=0.0

export PY_VER=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
export SCRIPT_DIR=${VIAME_INSTALL}/lib/python${PY_VER}/site-packages/viame/pytorch/siammask

source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline

rm -rf ${TRAIN_FOLDER}
mkdir -p ${TRAIN_FOLDER}

python -m torch.distributed.launch \
        --nproc_per_node=${GPU_COUNT} \
        ${SCRIPT_DIR}/siammask_trainer.py \
        -i ${DATA_FOLDER} \
        -s ${TRAIN_FOLDER} \
        -c ${VIAME_INSTALL}/configs/pipelines/models/siammask_training_config.yaml \
        --threshold ${THRESH} # --skip-crop
