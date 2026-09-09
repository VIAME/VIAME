#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

viame index add -l ingest_list.txt \
  --method frames -install ${VIAME_INSTALL}
