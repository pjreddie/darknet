#!/usr/bin/env bash

number_of_build_workers=8
bypass_vcpkg=true

if [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ "$1" == "gcc" ]]; then
    export CC="/usr/local/bin/gcc-8"
    export CXX="/usr/local/bin/g++-8"
  fi
  vcpkg_triplet="x64-osx"
else
  vcpkg_triplet="x64-linux"
fi

if [[ ! -z "${VCPKG_ROOT}" ]] && [ -d ${VCPKG_ROOT} ] && [ ! "$bypass_vcpkg" = true ]
then
  vcpkg_path="${VCPKG_ROOT}"
  vcpkg_define="-DCMAKE_TOOLCHAIN_FILE=${vcpkg_path}/scripts/buildsystems/vcpkg.cmake"
  vcpkg_triplet_define="-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet"
  echo "Found vcpkg in VCPKG_ROOT: ${vcpkg_path}"
elif [[ ! -z "${WORKSPACE}" ]] && [ -d ${WORKSPACE}/vcpkg ] && [ ! "$bypass_vcpkg" = true ]
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
