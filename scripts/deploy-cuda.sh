#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Unable to deploy CUDA on macOS, please wait for a future script update"
else
  if [[ $(cut -f2 <<< $(lsb_release -r)) == "18.04" ]]; then
    sudo apt-get update
    sudo apt-get install build-essential g++
    sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt-get update
    sudo apt-get dist-upgrade -y
    sudo apt-get install -y --no-install-recommends cuda-compiler-10-2 cuda-libraries-dev-10-2 cuda-driver-dev-10-2 cuda-cudart-dev-10-2 cuda-curand-dev-10-2
    sudo apt-get install -y --no-install-recommends libcudnn7-dev
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
  elif [[ $(cut -f2 <<< $(lsb_release -r)) == "20.04" ]]; then
    sudo apt-get update
    sudo apt-get install build-essential g++
    sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
    sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /"
    sudo apt-get update
    sudo apt-get dist-upgrade -y
    sudo apt-get install -y --no-install-recommends cuda-compiler-11-2 cuda-libraries-dev-11-2 cuda-driver-dev-11-2 cuda-cudart-dev-11-2
    sudo apt-get install -y --no-install-recommends libcudnn8-dev
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-11.2 /usr/local/cuda
  else
    echo "Unable to deploy CUDA on this Linux version, please wait for a future script update"
  fi
fi
