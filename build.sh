#!/usr/bin/env bash

## by default darknet is built with CC 3.0 if cuda is found. Uncomment another CC (only one!) if you prefer a different choice
#my_cuda_compute_model=75    #Compute capability for Tesla T4, RTX 2080
#my_cuda_compute_model=72    #Compute capability for Jetson Xavier
#my_cuda_compute_model=70    #Compute capability for Tesla V100
#my_cuda_compute_model=62    #Compute capability for Jetson TX2
#my_cuda_compute_model=61    #Compute capability for Tesla P40
#my_cuda_compute_model=60    #Compute capability for Tesla P100
#my_cuda_compute_model=53    #Compute capability for Jetson TX1
#my_cuda_compute_model=52    #Compute capability for Tesla M40/M60
#my_cuda_compute_model=37    #Compute capability for Tesla K80
#my_cuda_compute_model=35    #Compute capability for Tesla K20/K40
#my_cuda_compute_model=30    #Compute capability for Tesla K10, Quadro K4000

number_of_build_workers=8

if [[ "$OSTYPE" == "darwin"* ]]; then
  OpenCV_DIR="/usr/local/Cellar/opencv@3/3.4.5"
  additional_defines="-DOpenCV_DIR=${OpenCV_DIR}"
  if [[ "$1" == "gcc" ]]; then
    export CC="/usr/local/bin/gcc-8"
    export CXX="/usr/local/bin/g++-8"
  fi
fi

if [[ ! -z "$my_cuda_compute_model" ]]; then
  additional_build_setup="-DCUDA_COMPUTE_MODEL=${my_cuda_compute_model}"
fi

# RELEASE
mkdir -p build_release
cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release ${additional_defines} ${additional_build_setup}
cmake --build . --target install -- -j${number_of_build_workers}
#cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
rm -f DarknetConfig.cmake
rm -f DarknetConfigVersion.cmake
cd ..
cp cmake/Modules/*.cmake share/darknet

# DEBUG
mkdir -p build_debug
cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug ${additional_defines} ${additional_build_setup}
cmake --build . --target install -- -j${number_of_build_workers}
#cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
rm -f DarknetConfig.cmake
rm -f DarknetConfigVersion.cmake
cd ..
cp cmake/Modules/*.cmake share/darknet
