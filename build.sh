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
bypass_vcpkg=true

if [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ "$1" == "gcc" ]]; then
    export CC="/usr/local/bin/gcc-8"
    export CXX="/usr/local/bin/g++-8"
  fi
  vcpkg_triplet="x64-darwin"
else
  vcpkg_triplet="x64-linux"
fi

if [[ ! -z "$my_cuda_compute_model" ]]; then
  additional_build_setup="-DCUDA_COMPUTE_MODEL=${my_cuda_compute_model}"
fi

if [[ -d ${VCPKG_ROOT} ]] && [ ! "$bypass_vcpkg" = true ]
then
  vcpkg_path="${VCPKG_ROOT}"
  vcpkg_define="-DCMAKE_TOOLCHAIN_FILE=${vcpkg_path}/scripts/buildsystems/vcpkg.cmake"
  vcpkg_triplet_define="-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet"
  echo "Found vcpkg in VCPKG_ROOT: ${vcpkg_path}"
elif [ -d ${WORKSPACE}/vcpkg${vcpkg_fork} ] && [ ! "$bypass_vcpkg" = true ]
then
  vcpkg_path="${WORKSPACE}/vcpkg"
  vcpkg_define="-DCMAKE_TOOLCHAIN_FILE=${vcpkg_path}/scripts/buildsystems/vcpkg.cmake"
  vcpkg_triplet_define="-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet"
  echo "Found vcpkg in WORKSPACE/vcpkg: ${vcpkg_path}"
elif [ ! "$bypass_vcpkg" = true ]
then
  (>&2 echo "darknet is unsupported without vcpkg, use at your own risk!")
fi

## DEBUG
#mkdir -p build_debug
#cd build_debug
#cmake .. -DCMAKE_BUILD_TYPE=Debug ${vcpkg_define} ${vcpkg_triplet_define} ${additional_defines} ${additional_build_setup}
#cmake --build . --target install -- -j${number_of_build_workers}
##cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
#rm -f DarknetConfig.cmake
#rm -f DarknetConfigVersion.cmake
#cd ..
#cp cmake/Modules/*.cmake share/darknet/

# RELEASE
mkdir -p build_release
cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release ${vcpkg_define} ${vcpkg_triplet_define} ${additional_defines} ${additional_build_setup}
cmake --build . --target install -- -j${number_of_build_workers}
#cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
rm -f DarknetConfig.cmake
rm -f DarknetConfigVersion.cmake
cd ..
cp cmake/Modules/*.cmake share/darknet/
