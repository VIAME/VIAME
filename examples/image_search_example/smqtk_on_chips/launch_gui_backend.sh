#!/usr/bin/env bash
#
# Start mongo database some where with:
#	mongod --dbpath $PWD
# Where $PWD is some directory
#
source ../../../setup_viame.sh

runApplication -a IqrSearchDispatcher \
  -c configs/runApp.IqrSearchDispatcher.json \
  -tv
