#!/usr/bin/env bash

## enable or disable installed components

install_cuda=false
vcpkg_folder="."
temp_folder="./temp"

###########################

mkdir $temp_folder
cd $temp_folder

sudo apt-get install cmake git ninja-build build-essentials g++

if [ "$install_cuda" = true ] ; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Unable to provide CUDA on macOS"
  else
    # Download and install CUDA
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub
    sudo dpkg -i cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb
    sudo dpkg -i nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb
    sudo apt-get -y update
    sudo apt-get install -y --no-install-recommends cuda-compiler-10-0 cuda-libraries-dev-10-0 cuda-driver-dev-10-0 cuda-cudart-dev-10-0 cuda-cublas-dev-10-0 cuda-curand-dev-10-0
    sudo apt-get install -y --no-install-recommends libcudnn7-dev
    sudo ln -s /usr/local/cuda-10.0/lib64/stubs/libcuda.so /usr/local/cuda-10.0/lib64/stubs/libcuda.so.1

    export CUDACXX=/usr/local/cuda-10.0/bin/nvcc
    export CUDA_PATH=/usr/local/cuda-10.0
    export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0
    export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib64/stubs:${LD_LIBRARY_PATH}"

    features = "full"
  fi
else
  features = "opencv-base,weights,weights-train"
fi

rm -rf $temp_folder
cd ..
cd $vcpkg_folder
git clone https://github.com/Microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install darknet[${features}]
