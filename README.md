![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

# Darknet CrowdHuman

As part of my Final Year Project, I have trained the Yolo object detector on the Crowdhuman dataset. The goal was to be able to achieve fast detections on people and faces in crowds.

## CrowdHuman Dataset

From the CrowdHuman [website](http://www.crowdhuman.org/):

>CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

### Train/Validation
The CrowdHuman dataset can be downloaded from the [here](http://www.crowdhuman.org/download.html).

The training set is divided into 3 files, and are between 2-3GB zipped.
* ``CrowdHuman_train01.zip``
* ``CrowdHuman_train02.zip``
* ``CrowdHuman_train03.zip``

A validation set is also provided in ``CrowdHuman_val.zip``

Both the __training__ and __validation__ sets come with annotations.

* ``annotation_train.odgt``
* ``annotation_val.odgt``

#### Annotation Format
The annotations come in the ``.odtg`` format. Each line in the files is a JSON containing the annotations found in the referred image.

### Test
The test set is provided ``CrowdHuman_test.zip``. As far as I can tell, there are no labels for the test set.

## Training Darknet
Training on Darknet is never fun. There are about a million different tutorials on how to setup the files, what to do, what files to change. Many tutorials are out of date. I'm about to contribute to this mess. 

Note: This tutorial is not meant to be general 'how to'. It is how __I__ managed to train YOLO.

I trained the ``yolov3-tiny`` with ``2 classes`` on:
* Ubuntu 16.04 with Standard Darknet Setup (GPU, OPENCV, CUDNN)
* GTX 1050 Ti (4GB RAM)

### Setting up files
1. Download the 3 training zip files ``CrowdHuman_train0*.zip`` from the CrowdHuman website.
2. Extract all the files into the folder provided ``/darknet/crowdhuman_train``. You should have about __15000__ files in the folder.
3. Download the validation zip file ``CrowdHuman_val.zip``
4. Extract the validation set into ``/darknet/crowdhuman_val``. I think theres about __4370__ files in there.
5. Download the both ``annotation_train.odgt`` and ``annotation_val.odgt`` and place them in the main ``/darknet/`` folder.

### Crowdhuman to Darknet Format
CrowdHuman provides its annotations in the ``.odtg`` JSON format. Darknet does not like this. Darknet expects its annotations as such:

* Each image has a corresponding textfile containing the annotations.
    * Example: ``dog.jpg`` would have annotations in ``dog.txt``, in the same folder.

* Annotations in Darknet look like this:
    * ``<object-class> <x> <y> <width> <height>``
    * Each line in the textfile is of that format, and each line represents an object.
    * ``x, y`` is the centre of the bounding box.
    * ``width/height`` is from the centre of the box.
    * All these values need to be __scaled__ with respect to the size of the image.
        * ``x = xCentre / imgWidth``
        * ``y = yCentre / imgHeight``
        * ``width = widthBoundingBox / imgWidth``
        * ``height = heightBoundingBox / imgHeight``
    * All the values should be between 0 and 1.

#### Generate Annotations
I have written some files which convert and create the textfiles containing the annotations:
* ``crowdhuman_train_anno.py``
* ``crowdhuman_val_anno.py``

Simply run those two files from the terminal ``python crowdhuman_*_anno.py`` from this folder, and it will generate all the corresponding textfiles in the ``/darknet/crowhuman_train`` and ``/darknet/crowhuman_val`` directories.

#### Generate Image Filepaths
Darknet also needs another textfile which contains the paths to all the training and validation images. I have written some scripts to do this:

* ``generate_train_txt.py``
* ``generate_val_txt.py``

This generates two files ``train.txt`` and ``val.txt`` in the main ``/darknet/`` directory.

### Actually Training Darknet

Just run the command:

```
./darknet detector train cfg/yolo_crowdhuman.data cfg/yolov3-tiny-crowdhuman.cfg darknet53.conv.74
```

Where ``darknet53.conv.74`` is the initial weights which one can get from:

```
wget https://pjreddie.com/media/files/darknet53.conv.74
```

#### Training Tips & Tricks
* No GUI to save GPU Memory
    * Might run out of memory on your GPU, so a good hack is to just run the training without any GUI. I used the virtual terminals ``tty1``. (Or pressing CTRL + ALT + F1)
    * I killed the GUI by running ``sudo service lightdm stop``. This left me with just a terminal and I trained the network there.