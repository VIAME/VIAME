#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/launch_search_interface.py \
  -qp pipelines/query_retrieval_and_iqr.pipe
