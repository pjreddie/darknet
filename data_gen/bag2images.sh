#!/usr/bin/env bash

# Clear old data
num=`ls -l images/ | wc -l`
if [ $num -gt 1 ]
then
    echo "Emptying images directory..."
    rm images/*
fi

openLabelCount=`ls -l OpenLabeling/main/input/ | wc -l`
if [ $openLabelCount -ne 1 ]
then
    read -p "OpenLabeling directory currently contains images. Delete and empty the input & output directory? (y/n)" res
    case $res in
	[Yy]* ) rm OpenLabeling/main/input/*; rm -r OpenLabeling/main/output/*; openLabelCount=1;;
	* ) ;;
    esac
fi

# Get bag file ready to play
cd bag_file
num=`ls -1 | wc -l`
if [ $num -ne 1 ]
then
    echo "ERROR: Only one bag file allowed in the bag_file directory"
    exit
fi

orig_file_name=$(ls)
file_name_extension=${orig_file_name::-4}

echo "Decompressing "$orig_file_name
rosbag decompress $orig_file_name
echo "Completed decompression"

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

# Save the bag file as jpegs
echo "---------------------"
echo "Replaying rosbag now and generating images"
echo "---------------------"
cd ..
dir=`pwd`
file_path=`pwd`/bag_file/$orig_file_name
export_path=`pwd`/images/$file_name_extension

# launch the loading bar in the bkgnd
./scripts/loading_bar.py $bag_file_seconds &

# export images to a specific path
roslaunch launch/export.launch bag_file:=$file_path image_dir:=$export_path &> tmp.txt

# make sure everything's killed
rm tmp.txt
pkill python

echo "Finished Exporting images"

num=`ls -1 images/ | wc -l`
echo "Number of images: "$num
zip_name=${orig_file_name::-4}_unlabeled.zip

if [ $openLabelCount -eq 1 ]
then
    cp images/* OpenLabeling/main/input/
    echo "Images have been copied to OpenLabeling/main/input/, ready for labeling"
fi

# Zip the unlabeled images
mkdir -p zipped_images
cd images
zip -q $zip_name *
mv $zip_name ../zipped_images/
cd ..
rm images/*
echo "---------------------"
echo "Success, images have been zipped and are located in zipped_images/"$zip_name

echo "Deleting the uncompressed bag file for unneccessary space usage"
rm bag_file/$orig_file_name
mv bag_file/* bag_file/$orig_file_name

