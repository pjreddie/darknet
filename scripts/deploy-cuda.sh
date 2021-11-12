#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Unable to deploy CUDA on macOS, please wait for a future script update"
  exit 1
elif [[ $(cut -f2 <<< $(lsb_release -i)) == "Ubuntu" ]]; then
  distr_name="$(cut -f2 <<< $(lsb_release -i) | tr '[:upper:]' '[:lower:]')$(cut -f2 <<< $(lsb_release -r) | tr -d '.')"
else
  echo "Unable to deploy CUDA on this OS, please wait for a future script update"
  exit 3
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -f $script_dir/requested_cuda_version.sh ]; then
  source $script_dir/requested_cuda_version.sh
else
  echo "Unable to find requested_cuda_version.sh script"
  exit 1
fi

sudo apt-get update
sudo apt-get install build-essential g++
sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/cuda-$distr_name.pin
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/ /"
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/machine-learning/repos/$distr_name/x86_64/ /"
sudo apt-get update
sudo apt-get dist-upgrade -y
sudo apt-get install -y --no-install-recommends cuda-compiler-${CUDA_VERSION_DASHED} cuda-libraries-dev-${CUDA_VERSION_DASHED} cuda-driver-dev-${CUDA_VERSION_DASHED} cuda-cudart-dev-${CUDA_VERSION_DASHED}
sudo apt-get install -y --no-install-recommends libcudnn8-dev
sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda
