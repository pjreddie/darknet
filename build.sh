#!/usr/bin/env bash

number_of_build_workers=8
bypass_vcpkg=true
force_cpp_build=false

if [[ "$OSTYPE" == "darwin"* ]]; then
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
  additional_defines="-DBUILD_SHARED_LIBS=OFF"
elif [[ ! -z "${WORKSPACE}" ]] && [ -d ${WORKSPACE}/vcpkg ] && [ ! "$bypass_vcpkg" = true ]
then
  vcpkg_path="${WORKSPACE}/vcpkg"
  vcpkg_define="-DCMAKE_TOOLCHAIN_FILE=${vcpkg_path}/scripts/buildsystems/vcpkg.cmake"
  vcpkg_triplet_define="-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet"
  echo "Found vcpkg in WORKSPACE/vcpkg: ${vcpkg_path}"
  additional_defines="-DBUILD_SHARED_LIBS=OFF"
elif [ ! "$bypass_vcpkg" = true ]
then
  (>&2 echo "darknet is unsupported without vcpkg, use at your own risk!")
fi

if [ "$force_cpp_build" = true ]
then
  additional_build_setup="-DBUILD_AS_CPP:BOOL=TRUE"
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
