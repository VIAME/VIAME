#!/bin/bash

# Setup VIAME Paths (no need to run multiple times if you already ran it)

source ../../../setup_viame.sh

# Make directory for query KWAs
mkdir -p database/Queries

# Run GUI
this_dir=$(readlink -f $(dirname $BASH_SOURCE[0]))
viqui --add-layer ${this_dir}/configs/context_view_bluemarble_low_res.kst
