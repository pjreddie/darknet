#!/usr/bin/env bash

## enable or disable installed components

install_cuda=true

###########################

temp_folder="./temp"
mkdir -p $temp_folder
cd $temp_folder

sudo apt-get install cmake git ninja-build build-essential g++

if [ "$install_cuda" = true ] ; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Unable to provide CUDA on macOS"
  else
    # Download and install CUDA
    if [[ $(cut -f2 <<< $(lsb_release -r)) == "18.04" ]]; then
      wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
      sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
      sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
      wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
      sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
      sudo apt update
      sudo apt-get dist-upgrade -y
      sudo apt-get install -y --no-install-recommends cuda-compiler-10-2 cuda-libraries-dev-10-2 cuda-driver-dev-10-2 cuda-cudart-dev-10-2 cuda-curand-dev-10-2
      sudo apt-get install -y --no-install-recommends libcudnn7-dev
      sudo rm -rf /usr/local/cuda
      sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
      export PATH=/usr/local/cuda/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
      export CUDACXX=/usr/local/cuda/bin/nvcc
      export CUDA_PATH=/usr/local/cuda
      export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
      features="full"
    elif [[ $(cut -f2 <<< $(lsb_release -r)) == "20.04" ]]; then
      sudo apt update
      sudo apt-get dist-upgrade -y
      #sudo apt-get install -y --no-install-recommends nvidia-cuda-dev nvidia-cuda-toolkit
      sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
      sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
      sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
      sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /"
      sudo apt-get install -y --no-install-recommends cuda-compiler-11-2 cuda-libraries-dev-11-2 cuda-driver-dev-11-2 cuda-cudart-dev-11-2
      sudo apt-get install -y --no-install-recommends libcudnn8-dev
      sudo rm -rf /usr/local/cuda
      sudo ln -s /usr/local/cuda-11.2 /usr/local/cuda
      export PATH=/usr/local/cuda/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
      export CUDACXX=/usr/local/cuda/bin/nvcc
      export CUDA_PATH=/usr/local/cuda
      export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
      features="full"
    else
      echo "Unable to auto-install CUDA on this Linux OS"
      features="opencv-base"
    fi
  fi
else
  if [[ -v CUDA_PATH ]]; then
    features="full"
  else
    features="opencv-base"
  fi
fi

cd ..
rm -rf $temp_folder

if [[ ! -v VCPKG_ROOT ]]; then
  git clone https://github.com/microsoft/vcpkg
  cd vcpkg
  ./bootstrap-vcpkg.sh -disableMetrics
  export VCPKG_ROOT=$(pwd)
fi

$VCPKG_ROOT/vcpkg install darknet[${features}]

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Darknet installed in $VCPKG_ROOT/installed/x64-osx/tools/darknet"
else
  echo "Darknet installed in $VCPKG_ROOT/installed/x64-linux/tools/darknet"
fi
