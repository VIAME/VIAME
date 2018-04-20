#!/usr/bin/env sh
set -e
export FS_ROOT=/docker/data/flaskfs
cd ${FS_ROOT}/model
/root/caffe/build/tools/caffe "$@" > ${FS_ROOT}/algorithm/log/stdout_log.txt 2> ${FS_ROOT}/algorithm/log/stderr_log.txt

