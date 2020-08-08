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

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt update
    sudo apt-get install -y --no-install-recommends cuda-compiler-10-2 cuda-libraries-dev-10-2 cuda-driver-dev-10-2 cuda-cudart-dev-10-2 cuda-curand-dev-10-2
    sudo apt-get install -y --no-install-recommends libcudnn7-dev
    sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export CUDA_PATH=/usr/local/cuda
    export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

    features = "full"
  fi
else
  features = "opencv-base,weights,weights-train"
fi

rm -rf $temp_folder
cd ..
cd $vcpkg_folder
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
./vcpkg install darknet[${features}]
