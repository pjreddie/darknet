#!/usr/bin/env bash

# Get bag file ready to play
cd bag_file
num=`ls -1 | wc -l`
if [ $num -gt 1 ]
then
    echo "ERROR: Only one bag file allowed in the bag_file directory"
    exit
fi

orig_file_name=$(ls)
echo $orig_file_name

# echo "Decompressing bag file"
# rosbag decompress $orig_file_name
# echo "Completed decompression"

duration_str=`rosbag info $orig_file_name | grep duration`
#echo $duration_str
strindex() { 
  x="${1%%$2*}"
  [[ "$x" = "$1" ]] && echo -1 || echo "${#x}"
}
search_char='('
search_index=`strindex "$duration_str" "$search_char"`+1
new_str=${duration_str: $search_index}
bag_file_seconds=${new_str%??}
echo $bag_file_seconds

# Save the bag file as jpegs
echo "---------------------"
echo "Replaying rosbag now and generating images"
echo "---------------------"
cd ..
dir=`pwd`
file_path=`pwd`/bag_file/$orig_file_name
export_path=`pwd`/images/

./scripts/loading_bar.py $bag_file_seconds &

roslaunch launch/export.launch bag_file:=$file_path image_dir:=$export_path &> tmp.txt

rm tmp.txt
pkill python


echo "Finished Exporting images"

num=`ls -1 images/ | wc -l`
echo "Number of images: "$num
