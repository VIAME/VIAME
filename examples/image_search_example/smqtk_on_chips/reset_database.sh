rm -rf models/*
rm -rf tiles/*

mkdir -p models

cd models
nohup mongod --dbpath . &
cd ..
