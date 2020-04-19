#!/bin/sh

# Input locations and types
export INPUT_DIRECTORY=training_data_mouss
export ANNOTATION_TYPE=viame_csv

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Perform indexing operation required for SVM model train
python ${VIAME_INSTALL}/configs/process_video.py --init -d ${INPUT_DIRECTORY} \
  -p pipelines/index_default.svm.pipe -o database --build-index \
  -auto-detect-gt ${ANNOTATION_TYPE} -install ${VIAME_INSTALL}

# Perform actual SVM model generation
export SVM_TRAIN_IMPORT="import viame.arrows.smqtk.smqtk_train_svm_models as trainer"
export SVM_TRAIN_START="trainer.generate_svm_models()"

python -c "${SVM_TRAIN_IMPORT};${SVM_TRAIN_START}"
