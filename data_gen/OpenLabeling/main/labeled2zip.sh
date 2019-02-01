#!/usr/bin/env bash

# sort the data
./sort_labeled.sh

cp class_list.txt positives/
cp class_list.txt negatives/

cd positives
ls imgs >> image_list.txt

# calculate the name of the zip file
# We'll get an image file and extract the first part of it before the word image
img_name=`ls imgs |head -n 1`

strindex() { 
  x="${1%%$2*}"
  [[ "$x" = "$1" ]] && echo -1 || echo "${#x}"
}
search_char='i' # search for the 'i' in image
search_index=`strindex "$img_name" "$search_char"`-1 # take out the '-'
name=${img_name::$search_index}

zip -q -r ../$name.zip class_list.txt image_list.txt imgs labels

cd ..

# Negatives
cd negatives

ls imgs >> image_list.txt
zip -q -r ../negatives-$name.zip class_list.txt image_list.txt imgs labels
cd ..

# clean up
rm -r positives negatives

echo "Zip files created: "$name".zip and negatives-"$name".zip"
