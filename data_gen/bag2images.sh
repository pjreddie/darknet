#!/usr/bin/env bash
# This script will convert a compressed bag file into a zip file of unlabeled jpgs (max size 2k images until it creates another), it optionally loads the jpgs into OpenLabeling for training

if [ -z $1 ]
then
    echo "You must pass in a bag file"
    exit
fi

# Clear old data
mkdir -p images/
num=`ls -l images/ | wc -l`
if [ $num -gt 1 ]
then
    echo "Clearing images directory..."
    rm images/*
fi

orig_file_name=$1
file_name_root=${orig_file_name::-4}

# Zip will not overwrite another zip with the same name
# Check for the zip before and delete it
mkdir -p zipped_images
zip_exists=`ls zipped_images | grep $file_name_root`
if [ -n "$zip_exists" ]; then
    echo "Deleting previously generated "$orig_file_name
    rm zipped_images/$file_name_root*
fi

check_compressed=`rosbag info $orig_file_name | grep compression |grep none`
if [ -z "$check_compressed" ]; then
    echo "Decompressing "$orig_file_name
    rosbag decompress $orig_file_name
    echo "Completed decompression"
else
    echo "Bag file already decompressed"
    already_compressed=1    
fi

duration_str=`rosbag info $orig_file_name | grep duration`
strindex() { 
  x="${1%%$2*}"
  [[ "$x" = "$1" ]] && echo -1 || echo "${#x}"
}

# Find the time in minutes
search_char=':'
search_index=`strindex "$duration_str" "$search_char"`+2
new_str=${duration_str: $search_index}
search_char='s'
search_index=`strindex "$new_str" "$search_char"`
dur_mins=${new_str::$search_index}

# find the time in seconds
search_char='('
search_index=`strindex "$duration_str" "$search_char"`+1
new_str=${duration_str: $search_index}
bag_file_seconds=${new_str%??}
echo "Bag file duration "$dur_mins"s"

# get the image topic
topic_str=`rosbag info $orig_file_name | grep topics | grep sensor_msgs/Image`
image_topic=(`echo $topic_str | sed -r 's/topics: //g'`)

# Save the bag file as jpegs
echo "---------------------"
echo "Replaying rosbag now and generating images"
echo "---------------------"
dir=`pwd`
file_path=`pwd`/$orig_file_name
export_path=`pwd`/images/$file_name_root

# launch the loading bar in the bkgnd
./scripts/loading_bar.py $bag_file_seconds &

# export images to a specific path
roslaunch launch/export.launch bag_file:=$file_path image_dir:=$export_path image_topic:=$image_topic &> tmp.txt

# make sure everything's killed
rm tmp.txt
pkill python
echo "Finished Exporting images"

# Inform user about data generated
num_images=`ls -1 images/ | wc -l`
echo "Number of images: "$num_images
zip_name=${orig_file_name::-4}_unlabeled.zip

# Zip the unlabeled images
cd images
zip -q $zip_name *
mv $zip_name ../zipped_images/
cd ..
rm images/*
echo "---------------------"
echo "Success, images have been zipped and are located in zipped_images/"$zip_name

if [ -z $already_compressed ]
then
    echo "Deleting the uncompressed bag file for unneccessary space usage"
    rm $orig_file_name
    mv $file_name_root.orig.bag $orig_file_name
fi


