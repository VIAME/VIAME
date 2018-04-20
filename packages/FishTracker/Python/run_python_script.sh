#!/bin/bash
export FS_ROOT=/docker/data/flaskfs
python "$@" > ${FS_ROOT}/algorithm/log/stdout_log.txt 2> ${FS_ROOT}/algorithm/log/stderr_log.txt

