# docker run --gpus '"device=1"' -it --name cuda nvidia/cuda bash

apt-get update
# apt install git-all
apt-get install python3
apt-get install python
apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
apt install python3-pip
apt-get install -y libsm6 libxext6 libxrender-dev
apt-get update
