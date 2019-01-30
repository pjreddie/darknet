#!/usr/bin/env bash

cd OpenLabeling
python -mpip install -U pip
sudo python -mpip install -U -r requirements.txt

cd ..
mkdir bag_file
mkdir images

echo "Put a bag file in the bag_file directory and run ./bag2images.sh"
