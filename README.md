![Darknet Logo](https://github.com/prabindh/darknet/blob/master/arapaho/darknetcpplogo.png)

# Darknet-cpp

Darknet-cpp project is a bug-fixed and C++ compilable version of darknet, an open source neural network framework written in C and CUDA. Darknet-cpp builds on Linux, Windows and also tested on Mac by users.

**Features**

* Uses same code-base as original darknet (ie same .c files are used). Modification is done only for runtime bug-fixes, compile time fixes for c++, and the build system itself. For list of bugs fixed, refer to this thread - https://groups.google.com/forum/#!topic/darknet/4Hb159aZBbA, and https://github.com/prabindh/darknet/issues

* The Linux build system supports 3 targets - 
  * original darknet (with gcc compiler), 
  * darknet-cpp (with g++ compiler and Visual Studio compiler), and 
  * Shared library (libdarknet-cpp-shared.so)

* Can use bounding boxes directly from Euclid object labeller (https://github.com/prabindh/euclid)

* C++ API - arapaho, that works in conjunction with libdarknet-cpp-shared.so, and a test wrapper that can read images or video files, and show detected regions in a complete C++ application.

* darknet-cpp supports OpenCV3. Tested on Ubuntu 16.04 and windows, with CUDA 8.x

* Note: darknet-cpp requires a C++11 compiler for arapaho builds.

* Note - LSTM changes in mainline, are in the branch "merge_lstm_gru_jun17". Master always is stable.

**Usage**

Using the Makefile in the root directory of the darknet source repository,

 * `make darknet` - only darknet (original code), with OPENCV=0
 * `make darknet-cpp` - only the CPP version, with OPENCV=1
 * `make darknet-cpp-shared` - build the shared-lib version (without darknet.c calling wrapper), OPENCV=1
 * `make arapaho` - build arapaho and its test wrapper (from within arapaho folder)
 
**Steps to train (Yolov2)**

* Download latest commit of darknet-cpp, ex

`git clone https://github.com/prabindh/darknet`

* Create Yolo compatible training data-set. I use this to create Yolo compatible bounding box format file, and training list file. 

https://github.com/prabindh/euclid

This creates a training list file (train.txt) that will be needed in next step of training.

* Change the files per below:

  * yolo-voc.cfg - change line classes=20 to suit desired number of classes
  * yolo-voc.cfg - change the number of filters in the CONV layer above the region layer - (#classes + 4 + 1)*(5), where 4 is '#of coords', and 5 is 'num' in the cfg file.
  * voc.data - change line classes=20, and paths to training image list file
  * voc.names - number of lines must be equal the number of classes

* Place label-images corresponding to name of classes in data/labels, ex - data/labels/myclassname1.png

* Download http://pjreddie.com/media/files/darknet19_448.conv.23

* Train as below

  `./darknet-cpp detector train ./cfg/voc-myclasses.data ./cfg/yolo-myconfig.cfg darknet19_448.conv.23`

  * Atleast for the few initial iterations, observe the log output, and ensure all images are found and being used. After convergence, detection can be performed using standard steps.

* Testing with Arapaho C++ API for detection

  Arapaho needs the darknet-cpp shared library (.so file on Linux, .dll on Windows). This can be built as below on Linux.

  `make darknet-cpp-shared`

  On Windows port, the .dll is built by default.

  Refer the file https://github.com/prabindh/darknet/blob/master/arapaho/arapaho_readme.txt for more details.

# How to file issues

If there is a need to report an issue with the darknet-cpp port, use the link - https://github.com/prabindh/darknet/issues.

Information required for filing an issue:

  * Output of `git log --format="%H" -n 1`

  * Options enabled in Makefile (GPU,CUDNN)

  * If using Arapaho C++ wrapper, what options were used to build

  * Platform being used (OS version, GPU type, CUDA version, and OpenCV version)

# Darknet-cpp for Windows

Currently tested with VS2013, CUDA8.0 on Win10. 

The solution file requires the below repository.

https://github.com/prabindh/darknet-cpp-windows

The Windows port does not require any additional downloads (like pthreads), and builds the same darknet code-base for Windows, to generate the darknet.dll. Building the Arapaho C++ API and test wrapper, creates arapaho.exe, that works exactly the same way as arapaho on Linux.

# Darknet

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
