#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Unable to deploy CUDA on macOS, please wait for a future script update"
  exit 1
elif [[ $(cut -f2 <<< $(lsb_release -i)) == "Ubuntu" ]]; then
  distr_name="$(cut -f2 <<< $(lsb_release -i) | tr '[:upper:]' '[:lower:]')$(cut -f2 <<< $(lsb_release -r) | tr -d '.')"
else
  echo "Unable to deploy CUDA on this OS, please wait for a future script update"
  exit 2
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -f $script_dir/requested_cuda_version.sh ]; then
  source $script_dir/requested_cuda_version.sh
else
  echo "Unable to find requested_cuda_version.sh script"
  exit 3
fi

sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential g++
sudo apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg software-properties-common wget
sudo apt-get install -y --no-install-recommends zlib1g
sudo apt-get dist-upgrade -y
sudo apt-get install -y --no-install-recommends cuda-${CUDA_VERSION_DASHED}
sudo apt-get install -y --no-install-recommends libcudnn8
sudo apt-get install -y --no-install-recommends libcudnn8-dev

sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda

sudo apt-get clean
