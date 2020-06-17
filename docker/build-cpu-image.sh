#!/bin/bash

cur_dir=$(pwd | rev | cut -d"/" -f1 | rev)

if [[ "$cur_dir" == "docker" ]]; then
  cd ..
fi

if ! [[ -f "./docker/.env" ]]; then
    cp ./docker/.env.example ./docker/.env
fi

source ./docker/.env

docker build \
 --build-arg DOCKER_TIMEZONE=${DOCKER_TIMEZONE} \
 --build-arg DOCKER_APP_UID=${DOCKER_APP_UID} \
 --build-arg DOCKER_APP_GID=${DOCKER_APP_GID} \
 --build-arg DOCKER_APP_USERNAME=${DOCKER_APP_USERNAME} \
 --build-arg DOCKER_APP_PASSWORD=${DOCKER_APP_PASSWORD} \
 -f ./docker/Dockerfile.cpu \
 -t darknet:cpu-latest \
 - < ./docker/Dockerfile.cpu