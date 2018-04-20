#!/bin/bash
BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${BINDIR}/../local/lib/python2.7:${BINDIR}/../local/lib/python2.7/dist-packages:${BINDIR}/../PythonModule/build:${BINDIR}/../python:/root/caffe/python
export LD_LIBRARY_PATH=${BINDIR}/../local/lib:${BINDIR}/../local/lib64:/usr/local/cuda-8.0/lib64:/root/opencv/build/lib
export FS_ROOT=/docker/data/flaskfs
echo $LD_LIBRARY_PATH
python "$@" > ${FS_ROOT}/algorithm/log/stdout_log.txt 2> ${FS_ROOT}/algorithm/log/stderr_log.txt





