#! /bin/bash

docker_tag=$1

# Build girder server
docker build -t kitware/viame/girder-server:$docker_tag -f web/docker/girder_server.Dockerfile .

# Build girder worker base
docker build -t kitware/viame/girder-worker-base:$docker_tag -f web/docker/girder_worker_base.Dockerfile .

# Build girder worker w/ VIAME
docker build -t kitware/viame/viame-girder-worker:$docker_tag -f web/docker/viame_girder_worker.Dockerfile .


