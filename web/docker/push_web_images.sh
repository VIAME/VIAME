#! /bin/bash

# From VIAME repo root directory:
# './web/docker/push_web_images.sh <docker-image-tag>'

# Script pushes local docker images to kitware's dockerhub.com VIAME project

docker_tag=$1

docker login
docker push kitware/viame/girder-server:$docker_tag
docker push kitware/viame/girder-worker-base:$docker_tag
docker push kitware/viame/viame-girder-worker:$docker_tag
