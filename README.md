![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet-cpp#
Darknet-cpp project is a bug-fixed and C++ compilable version of darknet, an open source neural network framework written in C and CUDA.

**Features**

* Uses same code-base as original darknet (ie same .c files are used). Modification is done only for runtime bug-fixes, compile time fixes for c++, and the build system itself. For list of bugs fixed, refer to this thread - https://groups.google.com/forum/#!topic/darknet/4Hb159aZBbA, and https://github.com/prabindh/darknet/issues

* Build system supports 3 targets - 
  * original darknet (with gcc compiler), 
  * darknet-cpp (with g++ compiler), and 
  * Shared library (libdarknet-cpp-shared.so)

* Can use bounding boxes directly from Euclid object labeller (https://github.com/prabindh/euclid)

* Work in progress C++ API - arapaho

**Usage**

 * `make darknet` - only darknet (original code), with OPENCV=0
 * `make darknet-cpp` - only the CPP version, with OPENCV=1
 * `make darknet-cpp-shared` - build the shared-lib version (without darknet.c calling wrapper), OPENCV=1
 
 darknet-cpp version supports OpenCV3. Tested on Ubuntu 16.04 anad CUDA 8.x

**Steps to train (Yolov2)**

Download latest tag of darknet-cpp, ex

https://github.com/prabindh/darknet/tree/v3.76

1. Create Yolo compatible training data-set. I use this to create Yolo compatible bounding box format file, and training list file. 

https://github.com/prabindh/euclid

This creates a training list file that will be needed in next step.

2. Change 3 files per below:

  * yolo-voc.cfg - change line classes=20 to suit desired number of classes
  * yolo-voc.cfg - change the number of filters in the CONV layer above the region layer - #classes + #coords(4) + 1)*(NUM)
  * voc.data - change line classes=20, and paths to training image list file
  * voc.names - number of lines must be equal the number of classes

3. Place label-images corresponding to name of classes in data/labels, ex - data/labels/myclassname1.png

4. Download http://pjreddie.com/media/files/darknet19_448.conv.23

5. Train as below

  `./darknet-cpp detector train ./cfg/voc-myclasses.data ./cfg/yolo-myconfig.cfg darknet19_448.conv.23`

Atleast for the few initial iterations, observe the log output, and ensure all images are found and being used. After convergence, detection can be performed using standard steps.


#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
