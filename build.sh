#!/usr/bin/env bash

number_of_build_workers=8

if [[ "$OSTYPE" == "darwin"* && "$1" == "gcc" ]]; then
  export CC="/usr/local/bin/gcc-8"
  export CXX="/usr/local/bin/g++-8"
fi

rm -f uselib darknet

# CPU ONLY, DEBUG
mkdir -p build_debug
cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --target install -- -j${number_of_build_workers}
#cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
cd ..

# CPU ONLY, RELEASE
mkdir -p build_release
cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install -- -j${number_of_build_workers}
#cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
cd ..

# CUDA, DEBUG
mkdir -p build_debug_gpu
cd build_debug_gpu
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA:BOOL=TRUE
cmake --build . --target install -- -j${number_of_build_workers}
#cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
cd ..

# CUDA, RELEASE
mkdir -p build_release_gpu
cd build_release_gpu
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA:BOOL=TRUE
cmake --build . --target install -- -j${number_of_build_workers}
#cmake --build . --target install --parallel ${number_of_build_workers}  #valid only for CMake 3.12+
cd ..
