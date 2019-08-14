#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..
export DATA_FOLDER=data_folder/
export MODEL_FOLDER=siamrpn++_model/
export NUM_PROC=1

source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline

python -m torch.distributed.launch \
        --nproc_per_node=${NUM_PROC} \
    ${VIAME_INSTALL}/lib/python3.6/site-packages/pysot/viame/viame_train_tracker.py \
        -i ${VIAME_INSTALL}/examples/object_tracker_training/${DATA_FOLDER} \
        -s ${VIAME_INSTALL}/examples/object_tracker_training/${MODEL_FOLDER} \
        -c ${VIAME_INSTALL}/configs/pipelines/models/pysot_training_config.yaml \
        --threshold 0.0 # --skip-crop
