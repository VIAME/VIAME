#!/bin/bash
export FS_ROOT=/docker/data/flaskfs
mkdir ${FS_ROOT}/model/$1
sed '1s/.*/net: "'$1'\/train_val.prototxt"/' ${FS_ROOT}/model/fish_type/solver.prototxt > ${FS_ROOT}/model/$1/solver.prototxt
sed -i '11s/.*/max_iter: 10000/' ${FS_ROOT}/model/$1/solver.prototxt
sed -i '14s/.*/snapshot: 2000/' ${FS_ROOT}/model/$1/solver.prototxt
sed -i '15s/.*/snapshot_prefix: "snapshot\/'$1'"/' ${FS_ROOT}/model/$1/solver.prototxt
sed '16s/.*/    source: "'$1'\/train.txt"/' ${FS_ROOT}/model/fish_type/train_val.prototxt > ${FS_ROOT}/model/$1/train_val.prototxt
sed -i '36s/.*/    source: "'$1'\/test.txt"/' ${FS_ROOT}/model/$1/train_val.prototxt
sed -i '361s/.*/    num_output: '$2'/' ${FS_ROOT}/model/$1/train_val.prototxt
sed '327s/.*/    num_output: '$2'/' ${FS_ROOT}/model/fish_type/deploy.prototxt > ${FS_ROOT}/model/$1/deploy.prototxt
sed '8s/.*/        "solver": "'$1'\/solver.prototxt"/' ${FS_ROOT}/algorithm/config/fish_type_train.json > ${FS_ROOT}/algorithm/config/$1_train.json
sed '9s/.*/        "model_def": "model\/'$1'\/deploy.prototxt",/' ${FS_ROOT}/algorithm/config/prop_mbari_type.json > ${FS_ROOT}/algorithm/config/prop_$1.json
sed -i '18s/.*/        "model_type": "model\/'$1'\/'$1'.json",/' ${FS_ROOT}/algorithm/config/prop_$1.json
sed -i '20s/.*/        "pretrained_model": "model\/snapshot\/'$1'_iter_10000.caffemodel"/' ${FS_ROOT}/algorithm/config/prop_$1.json
sed '9s/.*/        "model_def": "model\/'$1'\/deploy.prototxt",/' ${FS_ROOT}/algorithm/config/class_mbari_type.json > ${FS_ROOT}/algorithm/config/class_$1.json
sed -i '18s/.*/        "model_type": "model\/'$1'\/'$1'.json",/' ${FS_ROOT}/algorithm/config/class_$1.json
sed -i '20s/.*/        "pretrained_model": "model\/snapshot\/'$1'_iter_10000.caffemodel"/' ${FS_ROOT}/algorithm/config/class_$1.json

