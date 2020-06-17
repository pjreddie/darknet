#!/bin/bash

cur_dir=$(pwd | rev | cut -d"/" -f1 | rev)

if [[ "$cur_dir" == "docker" ]]; then
  cd ..
fi

if ! [[ -f "./docker/.env" ]]; then
    cp ./docker/.env.example ./docker/.env
fi

source ./docker/.env

docker run \
--rm -it \
--privileged \
--net=host \
--ipc=host \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev/"${DOCKER_CAMERA_DEVICE}":/dev/video0 \
-v $(pwd):/home/darknet/app \
darknet:cpu-latest \
bash