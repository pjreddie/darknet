# Data Generation Directory

This directory helps generate the data for YOLO to train with by:
- Converting bag files to jpgs 
- Loading the jpgs into OpenLabeling
- Following all of the naming conventions of the YTS system to avoid conflicting data names in the training set
- splitting the data into positives and negatives
- Exporting unlabeled and labeled images to zip files with ease

## Getting Started
1 Move a bag file into the bag_file directory
- Note: it is expected this bag file is compressed, straight from the google drive
2 Generate images from the bag file
```
./bag2images
```
3 If you wish to train the bag files now, select yes when prompted to move the images to OpenLabeling
- Note if you select No, a zip file with the string "_unlabeled" appended to it will be created containing the jpgs
4 Change directories to prepare for labeling
```
cd OpenLabeling/main
```
5 Begin Labeling images
```
python main.py
```
6 Once labeling is done, we are ready to export the images and load them and their labels to the drive. Run the following to split the data into positive and negative examples and create zip files.
```
./labeled2zip
```
7 You will see 2 zip files. Please put the "negatives-" one in the YTS/Negatives/Unprocessed directory. Put the other one in the same directory as the original bag file used to generate the data.

### Labeling images from an *_unlabeled.zip file
1 Move the zip file into the zipped_images directory
2 $ ./label_zip
3 $ cd OpenLabeling/main
4 $ python main.py
