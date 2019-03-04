#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

wd=`pwd`
IFS=','

dataset="Human3"
class_id=0
w=480
h=640
num=1

while read -a line; do
filename=$(printf "$dataset/img/%04d.txt" $num)
#rm $filename.txt
echo "$class_id " > $filename
printf "%.5f " "$(($((line[0] + line[2]/2))  * 100000 / $w))e-5"   >> $filename
printf "%.5f " "$(($((line[1] + line[3]/2))  * 100000 / $h))e-5"   >> $filename
printf "%.5f " "$(($((line[2]))  * 100000 / $w))e-5"   >> $filename
printf "%.5f " "$(($((line[3]))  * 100000 / $h))e-5"   >> $filename
num=$((num + 1))
done < $dataset/groundtruth_rect.txt

echo "$dataset" > $dataset/otb.names
 

find $dataset/img -name \*.jpg > otb_train.txt