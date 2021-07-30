#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Unable to deploy CUDA on macOS, please wait for a future script update"
else
  if [[ $(cut -f2 <<< $(lsb_release -r)) == "18.04" ]]; then
    sudo apt-get update
    sudo apt-get install build-essential g++
    sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
    sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/ /"
    sudo apt-get update
    sudo apt-get dist-upgrade -y
    sudo apt-get install -y --no-install-recommends cuda-compiler-11-4 cuda-libraries-dev-11-4 cuda-driver-dev-11-4 cuda-cudart-dev-11-4
    sudo apt-get install -y --no-install-recommends libcudnn8-dev
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-11.4 /usr/local/cuda
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
    sudo apt-get install -y --no-install-recommends cuda-compiler-11-4 cuda-libraries-dev-11-4 cuda-driver-dev-11-4 cuda-cudart-dev-11-4
    sudo apt-get install -y --no-install-recommends libcudnn8-dev
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-11.4 /usr/local/cuda
  else
    echo "Unable to deploy CUDA on this Linux version, please wait for a future script update"
  fi
fi
