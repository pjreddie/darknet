#!/usr/bin/env bash

install_tools=false
bypass_driver_installation=false

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -InstallCUDA|--InstallCUDA)
    install_tools=true
    shift
    ;;
    -BypassDRIVER|--BypassDRIVER)
    bypass_driver_installation=true
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "This script is located in $script_dir"
cd $script_dir/..
temp_folder="./temp"
mkdir -p $temp_folder
cd $temp_folder

if [ "$install_tools" = true ] ; then
  $script_dir/deploy-cuda.sh
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Unable to provide tools on macOS, please wait for a future script update or do not put -InstallCUDA command line flag to continue"
  else
    if [[ $(cut -f2 <<< $(lsb_release -r)) == "18.04" ]]; then
      sudo apt-get update
      sudo apt-get install git ninja-build build-essential g++ nasm yasm
      sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
      wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
      sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
      wget -q https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
      sudo dpkg -i packages-microsoft-prod.deb
      sudo add-apt-repository universe
      sudo apt-get update
      sudo apt-get dist-upgrade -y
      sudo apt-get install -y cmake
      sudo apt-get install -y powershell
      if [ "$bypass_driver_installation" = true ] ; then
        sudo ln -s /usr/local/cuda-11.4/lib64/stubs/libcuda.so /usr/local/cuda-11.4/lib64/stubs/libcuda.so.1
        sudo ln -s /usr/local/cuda-11.4/lib64/stubs/libcuda.so /usr/local/cuda-11.4/lib64/libcuda.so.1
        sudo ln -s /usr/local/cuda-11.4/lib64/stubs/libcuda.so /usr/local/cuda-11.4/lib64/libcuda.so
      fi
      export PATH=/usr/local/cuda/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
      export CUDACXX=/usr/local/cuda/bin/nvcc
      export CUDA_PATH=/usr/local/cuda
      export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
      cuda_is_available=true
    elif [[ $(cut -f2 <<< $(lsb_release -r)) == "20.04" ]]; then
      sudo apt-get update
      sudo apt-get install git ninja-build build-essential g++ nasm yasm
      sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
      wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
      sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
      wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
      sudo dpkg -i packages-microsoft-prod.deb
      sudo add-apt-repository universe
      sudo apt-get update
      sudo apt-get dist-upgrade -y
      sudo apt-get install -y cmake
      sudo apt-get install -y powershell
      if [ "$bypass_driver_installation" = true ] ; then
        sudo ln -s /usr/local/cuda-11.4/lib64/stubs/libcuda.so /usr/local/cuda-11.4/lib64/stubs/libcuda.so.1
        sudo ln -s /usr/local/cuda-11.4/lib64/stubs/libcuda.so /usr/local/cuda-11.4/lib64/libcuda.so.1
        sudo ln -s /usr/local/cuda-11.4/lib64/stubs/libcuda.so /usr/local/cuda-11.4/lib64/libcuda.so
      fi
      export PATH=/usr/local/cuda/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
      export CUDACXX=/usr/local/cuda/bin/nvcc
      export CUDA_PATH=/usr/local/cuda
      export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
      cuda_is_available=true
    else
      echo "Unable to provide tools on macOS, please wait for a future script update or do not put -InstallCUDA command line flag to continue"
    fi
  fi
fi

cd ..
rm -rf "$temp_folder"

if [[ -v CUDA_PATH ]]; then
  ./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN -DisableInteractive -DoNotUpdateDARKNET
  #./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN -EnableOPENCV_CUDA -DisableInteractive -DoNotUpdateDARKNET
else
  ./build.ps1 -UseVCPKG -EnableOPENCV -DisableInteractive -DoNotUpdateDARKNET
fi
