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

# Save the bag file as jpegs
echo "---------------------"
echo "Replaying rosbag now and generating images"
echo "---------------------"
cd ..
dir=`pwd`
file_path=`pwd`/bag_file/$orig_file_name
export_path=`pwd`/images/

roslaunch launch/export.launch bag_file:=$file_path image_dir:=$export_path &> tmp.txt

rm tmp.txt

echo "Finished Exporting images"

num=`ls -1 images/ | wc -l`
echo "Number of images: "$num
