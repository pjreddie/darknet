# Data Generation Directory

This directory helps generate the data for YOLO to train with by:
- Converting bag files to jpgs 
- Loading the jpgs into OpenLabeling
- Following all of the naming conventions of the YTS system to avoid conflicting data names in the training set
- splitting the data into positives and negatives
- Exporting unlabeled and labeled images to zip files with ease

## Getting Started
0. (Only need to do once) install needed pkgs for OpenLabeling:
```
./install.sh
```
NOTE: You'll need pip installed to install the needed pkgs
1. Move a bag file into the data_gen directory
- Note: it is expected the .bag file is compressed
2. Generate a zip archive of jpgs from the bag file by running the following script and passing in the name of the bag file
```
./bag2images ___.bag
```
3. A zip file ending in "_unlabeled.zip" will have been created containing the jpgs. From there see the section below titled "Labeling images from an *_unlabeled.zip file." Run the following command to unzip and load the unlabeled images into OpenLabeling:
```
./label_zip
```
4. To start labeling: change directories: 
```
cd OpenLabeling/main
```
5. Begin Labeling images. Take note of the keystrokes and ability to undo boxes by following the README directions in the OpenLabeling directory.
```
python main.py
```
6. Once labeling is done, we are ready to export the images and load them and their labels to the drive. Run the following to split the data into positive and negative examples and create zip files.
```
./labeled2zip
```
7. You will see two zip files. On the google drive, put the "negatives-" zip in the YTS/Negatives/Unprocessed directory. Put the other one in the same directory as the original bag file used to generate the data.

### Labeling images from an *_unlabeled.zip file
1. Move the zip file into the zipped_images directory
```
./label_zip
```
2. Continue with steps 4-7) above

