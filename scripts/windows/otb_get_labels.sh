#!/bin/bash
 
dataset=$1
w=$2
h=$3

# Parameters: Human3, CarScale, Human6, Biker
#w=480
#h=640
IFS=','
  
export LC_NUMERIC="en_US.UTF-8"

wd=`pwd`
dataset_path="data/$dataset"

class_id=0
num=1

mkdir data
wget http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/$dataset.zip

unzip -o $dataset.zip -d data

sed -i.bak 's/\o11/,/g' $dataset_path/groundtruth_rect.txt
sed -i.bak 's/\o11/,/g' $dataset_path/groundtruth_rect.txt
dos2unix $dataset_path/groundtruth_rect.txt

while read -r left right width height; do
filename=$(printf "$dataset_path/img/%04d.txt" $num)
#rm $filename.txt
echo "$class_id " > $filename
printf "%.5f " "$(($((left + width/2))  * 100000 / $w))e-5"   >> $filename
printf "%.5f " "$(($((right + height/2))  * 100000 / $h))e-5"   >> $filename
printf "%.5f " "$(($((width))  * 100000 / $w))e-5"   >> $filename
printf "%.5f " "$(($((height))  * 100000 / $h))e-5"   >> $filename
num=$((num + 1))
done < $dataset_path/groundtruth_rect.txt

echo "$dataset" > $dataset_path/otb.names
 

find $dataset_path/img -name \*.jpg > data/$dataset/train.txt

echo "classes = 1" > data/otb_$dataset.data
echo "train = data/$dataset/train.txt" >> data/otb_$dataset.data
echo "valid = data/$dataset/train.txt" >> data/otb_$dataset.data
echo "names = $dataset_path/otb.names" >> data/otb_$dataset.data
echo "backup = backup/" >> data/otb_$dataset.data
echo "results= results/" >> data/otb_$dataset.data