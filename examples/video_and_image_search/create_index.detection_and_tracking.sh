#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# To change this script to process a directory of videos, as opposed to images
# change "-l ingest_list.txt" to "-d videos" if videos is a directory with videos.
# Add "--backend postgres" to store the index in an embedded database instead of files.

viame index add -l ingest_list.txt \
  --method tracking -install ${VIAME_INSTALL}
