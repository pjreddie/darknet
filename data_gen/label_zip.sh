#!/usr/bin/env bash

num=`ls -l zipped_images/ | wc -l`
if [ $num -lt 2 ]
then
    echo "No zips located in zipped_images. Exiting"
    exit
fi

num=`ls -l OpenLabeling/main/input/ | wc -l`
if [ $num -ne 1 ]
then
    echo "OpenLabeling directory currently contains images. Please enter OpenLabeling/main and run labeled2zip.sh to save and clear these images if they are finished being labeled."
    exit
fi

cd zipped_images
zip_arr=(*.zip)
index=0
echo "Please select a zip file to move to OpenLabeling"
for i in ${zip_arr[@]}; do
    echo "["$index"] : "$i
    index=$((index+1))
done
read res

echo "Selected "${zip_arr[$res]}

cp ${zip_arr[$res]} ../OpenLabeling/main/input/
cd ../OpenLabeling/main/input
unzip ${zip_arr[$res]}
rm ${zip_arr[$res]}
cd ../../../
echo ${zip_arr[$res]}" unzipped and loaded into OpenLabeling."
