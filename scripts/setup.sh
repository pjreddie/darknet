#!/usr/bin/env bash

install_tools=false
install_cuda=false
bypass_driver_installation=false

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -InstallCUDA|--InstallCUDA)
    install_cuda=true
    shift
    ;;
    -InstallTOOLS|--InstallTOOLS)
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

if [ -f $script_dir/requested_cuda_version.sh ]; then
  echo "Loading $script_dir/requested_cuda_version.sh"
  source $script_dir/requested_cuda_version.sh
else
  echo "Unable to find requested_cuda_version.sh script"
  exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  if [ "$install_cuda" = true ] ; then
    echo "Unable to install CUDA on macOS, please wait for a future script update or do not put -InstallCUDA command line flag to continue"
    exit 2
  fi
  if [ "$install_tools" = true ] ; then
    echo "Unable to provide tools on macOS, please wait for a future script update or do not put -InstallTOOLS command line flag to continue"
    exit 3
  fi
elif [[ $(cut -f2 <<< $(lsb_release -i)) == "Ubuntu" ]]; then
  echo "Running in $(cut -f2 <<< $(lsb_release -i))"
  echo "InstallCUDA: $install_cuda"
  echo "InstallTOOLS: $install_tools"
  if [ "$install_cuda" = true ] ; then
    echo "Running $script_dir/deploy-cuda.sh"
    $script_dir/deploy-cuda.sh
    if [ "$bypass_driver_installation" = true ] ; then
      sudo ln -s /usr/local/cuda-${CUDA_VERSION}/lib64/stubs/libcuda.so /usr/local/cuda-${CUDA_VERSION}/lib64/stubs/libcuda.so.1
      sudo ln -s /usr/local/cuda-${CUDA_VERSION}/lib64/stubs/libcuda.so /usr/local/cuda-${CUDA_VERSION}/lib64/libcuda.so.1
      sudo ln -s /usr/local/cuda-${CUDA_VERSION}/lib64/stubs/libcuda.so /usr/local/cuda-${CUDA_VERSION}/lib64/libcuda.so
    fi
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export CUDA_PATH=/usr/local/cuda
    export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
  fi
  if [ "$install_tools" = true ] ; then
    echo "Installing tools"
    sudo apt-get update
    sudo apt-get install -y git ninja-build build-essential g++ nasm yasm gperf
    sudo apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget
    sudo apt-get install -y libgles2-mesa-dev libx11-dev libxft-dev libxext-dev libxrandr-dev libxi-dev libxcursor-dev libxdamage-dev libxinerama-dev
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(cut -f2 <<< $(lsb_release -c)) main"
    wget -q https://packages.microsoft.com/config/ubuntu/$(cut -f2 <<< $(lsb_release -r))/packages-microsoft-prod.deb
    sudo dpkg -i packages-microsoft-prod.deb
    sudo add-apt-repository universe
    sudo apt-get update
    sudo apt-get dist-upgrade -y
    sudo apt-get install -y cmake
    sudo apt-get install -y powershell
    sudo apt-get install -y curl zip unzip tar
    sudo apt-get install -y pkg-config autoconf libtool bison
  fi
else
  if [ "$install_cuda" = true ] ; then
    echo "Unable to install CUDA on this OS, please wait for a future script update or do not put -InstallCUDA command line flag to continue"
    exit 4
  fi
  if [ "$install_tools" = true ] ; then
    echo "Unable to install tools on this OS, please wait for a future script update or do not put -InstallTOOLS command line flag to continue"
    exit 5
  fi
fi

cd ..
rm -rf "$temp_folder"
echo "Building darknet"
if [[ -v CUDA_PATH ]]; then
  ./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN -DisableInteractive -DoNotUpdateTOOL
  #./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN -EnableOPENCV_CUDA -DisableInteractive -DoNotUpdateTOOL
else
  ./build.ps1 -UseVCPKG -EnableOPENCV -DisableInteractive -DoNotUpdateTOOL
fi
