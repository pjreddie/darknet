# docker run --gpus '"device=1"' -it --name cuda nvidia/cuda bash

apt-get update
# apt install git-all
apt-get install -y python3
apt-get install -y python
apt install -y curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
apt install -y python3-pip
apt-get install -y libsm6 libxext6 libxrender-dev
pip3 install -r ../requirements.txt
apt install -y vim
pip3 install image-slicer
apt-get update
#apt-get upgrade
apt install libopencv-dev
