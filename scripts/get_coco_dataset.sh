#!/bin/bash

# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

# Download Images
wget -c https://pjreddie.com/media/files/train2014.zip --no-check-certificate
wget -c https://pjreddie.com/media/files/val2014.zip --no-check-certificate

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip

cd ..

# Download COCO Metadata
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip --no-check-certificate
wget -c https://pjreddie.com/media/files/coco/5k.part --no-check-certificate
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part --no-check-certificate
wget -c https://pjreddie.com/media/files/coco/labels.tgz --no-check-certificate
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

