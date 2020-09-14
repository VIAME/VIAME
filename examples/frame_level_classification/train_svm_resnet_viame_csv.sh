#!/bin/sh

# Input locations and types

export INPUT_DIRECTORY=training_data
export ANNOTATION_TYPE=viame_csv
export CONFIG_FILE=pipelines/index_full_frame.svm.pipe
#export CONFIG_FILE=pipelines/index_full_frame.svm.annot_only.pipe

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Perform indexing operation required for SVM model train

python ${VIAME_INSTALL}/configs/process_video.py --init -d ${INPUT_DIRECTORY} \
  -p ${CONFIG_FILE} -o database --build-index \
  -auto-detect-gt ${ANNOTATION_TYPE} -install ${VIAME_INSTALL}

# Perform actual SVM model generation

export SVM_TRAIN_IMPORT="import viame.arrows.smqtk.smqtk_train_svm_models as trainer"
export SVM_TRAIN_START="trainer.generate_svm_models()"

python -c "${SVM_TRAIN_IMPORT};${SVM_TRAIN_START}"
