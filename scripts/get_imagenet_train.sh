#!/bin/bash

wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
mkdir -p ILSVRC2012_img_train
tar --force-local -xf ILSVRC2012_img_train.tar -C ILSVRC2012_img_train

wd=`pwd`

for f in ILSVRC2012_img_train/*.tar;
do
name=$(echo "$f" | cut -f 1 -d '.')
mkdir "${wd}/${name}"
tar --force-local -xf "${wd}/${f}" -C "${wd}/${name}"
done

find "${wd}/ILSVRC2012_img_train" -name \*.JPEG > imagenet1k.train.list

