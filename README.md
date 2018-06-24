![Darknet Logo](https://github.com/prabindh/darknet/blob/master/arapaho/darknetcpplogo.png)

# Darknet-cpp

Darknet-cpp project is a bug-fixed and C++ compilable version of darknet (including Yolov3 and v2), an open source neural network framework written in C and CUDA. Darknet-cpp builds on Linux, Windows and also tested on Mac by users.

**Prebuilt binaries for evaluation now available**

* Prebuilt binaries are provided for evaluation purposes.
  * Tegra TK1 (CPU, CUDA, CUDA + CUDNN) - Yolov2 only
  * Windows x64 (CUDA + CUDNN) - Yolov3
  * Linux x64 (CUDA + CUDNN) - Yolov3
  * Darwin Mac x64 (CUDA + CUDNN) - Yolov3
  
* Download the binaries from yolo-bins 
  * https://github.com/prabindh/yolo-bins

**Features of darknet-cpp**

* Uses same source code-base as original darknet (ie same .c files are used). Modification is done only for runtime bug-fixes, compile time fixes for c++, and the build system itself. For list of bugs fixed, refer to this thread - https://groups.google.com/forum/#!topic/darknet/4Hb159aZBbA, and https://github.com/prabindh/darknet/issues

* The Linux build system supports 3 targets - 
  * original darknet (with gcc compiler), 
  * darknet-cpp (with g++ compiler and Visual Studio compiler), and 
  * Shared library (libdarknet-cpp-shared.so)

* Can use bounding boxes directly from Euclid object labeller (https://github.com/prabindh/euclid)

* C++ API - arapaho, that works in conjunction with libdarknet-cpp-shared.so, and a test wrapper that can read images or video files, and show detected regions in a complete C++ application.

* darknet-cpp supports OpenCV3. Tested on Ubuntu 16.04 and windows, with CUDA 8.x

* Note: darknet-cpp requires a C++11 compiler for darknet-cpp, and arapaho builds.

**Usage**

Using the Makefile in the root directory of the darknet source repository,

 * `make darknet` - only darknet (original code), with OPENCV=0
 * `make darknet-cpp` - only the CPP version, with OPENCV=1
 * `make darknet-cpp-shared` - build the shared-lib version (without darknet.c calling wrapper), OPENCV=1
 * `make arapaho` - build arapaho and its test wrapper (from within arapaho folder)

# Steps to test (Yolov3)

    After performing `make darknet-cpp`, the executable `darknet-cpp` is generated. Run the below command to recognise the provided data objects.

    ./darknet-cpp detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
    layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   256 x 256 x   3   ->   256 x 256 x  32  0.113 BFLOPs
    1 conv     64  3 x 3 / 2   256 x 256 x  32   ->   128 x 128 x  64  0.604 BFLOPs
    2 conv     32  1 x 1 / 1   128 x 128 x  64   ->   128 x 128 x  32  0.067 BFLOPs
    3 conv     64  3 x 3 / 1   128 x 128 x  32   ->   128 x 128 x  64  0.604 BFLOPs
    4 res    1                 128 x 128 x  64   ->   128 x 128 x  64
    5 conv    128  3 x 3 / 2   128 x 128 x  64   ->    64 x  64 x 128  0.604 BFLOPs
    6 conv     64  1 x 1 / 1    64 x  64 x 128   ->    64 x  64 x  64  0.067 BFLOPs
    7 conv    128  3 x 3 / 1    64 x  64 x  64   ->    64 x  64 x 128  0.604 BFLOPs
    8 res    5                  64 x  64 x 128   ->    64 x  64 x 128
    9 conv     64  1 x 1 / 1    64 x  64 x 128   ->    64 x  64 x  64  0.067 BFLOPs
    10 conv    128  3 x 3 / 1    64 x  64 x  64   ->    64 x  64 x 128  0.604 BFLOPs
    11 res    8                  64 x  64 x 128   ->    64 x  64 x 128
    12 conv    256  3 x 3 / 2    64 x  64 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    13 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    14 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    15 res   12                  32 x  32 x 256   ->    32 x  32 x 256
    16 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    17 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    18 res   15                  32 x  32 x 256   ->    32 x  32 x 256
    19 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    20 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    21 res   18                  32 x  32 x 256   ->    32 x  32 x 256
    22 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    23 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    24 res   21                  32 x  32 x 256   ->    32 x  32 x 256
    25 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    26 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    27 res   24                  32 x  32 x 256   ->    32 x  32 x 256
    28 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    29 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    30 res   27                  32 x  32 x 256   ->    32 x  32 x 256
    31 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    32 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    33 res   30                  32 x  32 x 256   ->    32 x  32 x 256
    34 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    35 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    36 res   33                  32 x  32 x 256   ->    32 x  32 x 256
    37 conv    512  3 x 3 / 2    32 x  32 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    38 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    39 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    40 res   37                  16 x  16 x 512   ->    16 x  16 x 512
    41 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    42 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    43 res   40                  16 x  16 x 512   ->    16 x  16 x 512
    44 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    45 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    46 res   43                  16 x  16 x 512   ->    16 x  16 x 512
    47 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    48 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    49 res   46                  16 x  16 x 512   ->    16 x  16 x 512
    50 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    51 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    52 res   49                  16 x  16 x 512   ->    16 x  16 x 512
    53 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    54 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    55 res   52                  16 x  16 x 512   ->    16 x  16 x 512
    56 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    57 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    58 res   55                  16 x  16 x 512   ->    16 x  16 x 512
    59 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    60 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    61 res   58                  16 x  16 x 512   ->    16 x  16 x 512
    62 conv   1024  3 x 3 / 2    16 x  16 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    63 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512  0.067 BFLOPs
    64 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    65 res   62                   8 x   8 x1024   ->     8 x   8 x1024
    66 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512  0.067 BFLOPs
    67 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    68 res   65                   8 x   8 x1024   ->     8 x   8 x1024
    69 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512  0.067 BFLOPs
    70 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    71 res   68                   8 x   8 x1024   ->     8 x   8 x1024
    72 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512  0.067 BFLOPs
    73 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    74 res   71                   8 x   8 x1024   ->     8 x   8 x1024
    75 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512  0.067 BFLOPs
    76 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    77 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512  0.067 BFLOPs
    78 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    79 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512  0.067 BFLOPs
    80 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.604 BFLOPs
    81 conv    255  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 255  0.033 BFLOPs
    82 detection
    83 route  79
    84 conv    256  1 x 1 / 1     8 x   8 x 512   ->     8 x   8 x 256  0.017 BFLOPs
    85 upsample            2x     8 x   8 x 256   ->    16 x  16 x 256
    86 route  85 61
    87 conv    256  1 x 1 / 1    16 x  16 x 768   ->    16 x  16 x 256  0.101 BFLOPs
    88 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    89 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    90 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    91 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256  0.067 BFLOPs
    92 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.604 BFLOPs
    93 conv    255  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 255  0.067 BFLOPs
    94 detection
    95 route  91
    96 conv    128  1 x 1 / 1    16 x  16 x 256   ->    16 x  16 x 128  0.017 BFLOPs
    97 upsample            2x    16 x  16 x 128   ->    32 x  32 x 128
    98 route  97 36
    99 conv    128  1 x 1 / 1    32 x  32 x 384   ->    32 x  32 x 128  0.101 BFLOPs
    100 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    101 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    102 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    103 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128  0.067 BFLOPs
    104 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.604 BFLOPs
    105 conv    255  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 255  0.134 BFLOPs
    106 detection
    Loading weights from yolov3.weights...Done!
    data/dog.jpg: Predicted in 0.066842 seconds.
    dog: 100%
    truck: 51%
    car: 77%
    bicycle: 74%
 
# Steps to train (Yolov2, tag v5.3)**

This section applies to git tag v5.3 and earlier. ie, Yolov2.

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

  On Windows, the .dll is built by default. The windows build requires the .sln files from (https://github.com/prabindh/darknet-cpp-windows)

  Refer the file https://github.com/prabindh/darknet/blob/master/arapaho/arapaho_readme.txt for more details on running Arapaho.

# How to file issues

If there is a need to report an issue with the darknet-cpp port, use the link - https://github.com/prabindh/darknet/issues.

Information required for filing an issue:

  * Output of `git log --format="%H" -n 1`

  * Options enabled in Makefile (GPU,CUDNN)

  * If using Arapaho C++ wrapper, what options were used to build

  * Platform being used (OS version, GPU type, CUDA version, and OpenCV version)

# Darknet-cpp for Windows

Currently tested with VS2015, CUDA8.0 on Win10 upto RS5. 

The solution file requires the below repository.

https://github.com/prabindh/darknet-cpp-windows

The Windows port does not require any additional downloads (like pthreads), and builds the same darknet code-base for Windows, to generate the darknet.dll. Building the Arapaho C++ API and test wrapper, creates arapaho.exe, that works exactly the same way as arapaho on Linux.

# Darknet

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
