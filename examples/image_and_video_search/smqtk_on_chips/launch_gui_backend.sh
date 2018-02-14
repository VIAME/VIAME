#!/usr/bin/env bash
#
# Start mongo database some where with:
#	mongod --dbpath $PWD
# Where $PWD is some directory
#
source ../../../setup_viame.sh

runApplication -a IqrSearchApp \
  -c configs/run_app.iqr_dispatcher.json \
  -tv
