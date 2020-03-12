#! /bin/bash

docker_tag=$1

docker login
docker push kitware/viame/girder-server:$docker_tag
docker push kitware/viame/girder-worker-base:$docker_tag
docker push kitware/viame/viame-girder-worker:$docker_tag
