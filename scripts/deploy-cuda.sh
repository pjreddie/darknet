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

sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/ /"
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/machine-learning/repos/$distr_name/x86_64/ /"
wget https://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/cuda-$distr_name.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/$distr_name/x86_64/ /"
sudo apt-get update
sudo apt-get install build-essential g++
sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
sudo apt-get install zlib1g
sudo apt-get dist-upgrade -y
sudo apt-get install -y --no-install-recommends cuda-${CUDA_VERSION_DASHED}
#sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
#sudo apt-get install libcudnn8-dev=${cudnn_version}-1+cuda${CUDA_VERSION}
sudo apt-get install libcudnn8
sudo apt-get install libcudnn8-dev

sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda
