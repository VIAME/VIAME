#! /bin/bash

set -x

export LC_ALL=en_US.utf8

export PATH=$PATH:/root/anaconda3/bin
source /root/anaconda3/bin/activate

pip install girder-client

cd /viame/build/
mv install viame
rm VIAME-v1.0.0-Linux-64Bit.tar.gz ||:
tar -zcvf VIAME-v1.0.0-Linux-64Bit.tar.gz  viame

girder-client --api-url https://data.kitware.com/api/v1 --api-key [INSERT-KEY] upload [INSERT-FIXED-HASH] VIAME-v1.0.0-Linux-64Bit.tar.gz  --reuse
