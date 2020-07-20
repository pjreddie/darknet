#!/bin/bash

cur_dir=$(pwd | rev | cut -d"/" -f1 | rev)

if [[ "$cur_dir" == "docker" ]]; then
  cd ..
fi

./docker/build-cpu-image.sh && ./docker/build-gpu-image.sh
