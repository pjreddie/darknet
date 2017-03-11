Arapaho is a C++ API for integrating darknet into C++ applications. Arapaho exposes 3 APIs, starting from tag 4.0 onwards.

Refer arapaho.hpp,

		bool Setup(ArapahoV2Params & p,
			int & expectedWidth,
			int & expectedHeight);

		bool Detect(
			ArapahoV2ImageBuff & imageBuff,
			float thresh,
			float hier_thresh,
			int & objectCount);

		bool GetBoxes(box* outBoxes, int boxCount);
		

Steps to build arapaho:

1. Build the darknet shared library

$ cd darknet
$ make darknet-cpp-shared

This will generate the shared library, libdarknet-cpp-shared.so. Copy this into arapaho folder

2. Copy the weights, cfg, data and image files for detection into the arapaho folder

$ cd arapaho
$ ls input*
input.cfg  input.data  input.jpg  input.weights

3. Build arapaho test app (including the arapaho core) for detection, using below command


g++ test.cpp arapaho.cpp -DGPU -DCUDNN -I../src/ -I/usr/local/cuda/include/ -L./ -ldarknet-cpp-shared -L/usr/local/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_viz -lopencv_core -lopencv_cudev -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -std=c++11

NOTE: GPU and CUDNN flags have to match what was specified in the build of darknet library, in step 1.

4. Run arapaho

$ ./a.out

This will generate the below output from the test application. Using GetBoxes() API, this can be easily integrated into any C++ application.

./a.out 
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
    1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
    2 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64
    3 max          2 x 2 / 2   208 x 208 x  64   ->   104 x 104 x  64
    4 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
    5 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64
    6 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
    7 max          2 x 2 / 2   104 x 104 x 128   ->    52 x  52 x 128
    8 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
    9 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
   10 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
   11 max          2 x 2 / 2    52 x  52 x 256   ->    26 x  26 x 256
   12 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   13 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
   14 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   15 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
   16 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   17 max          2 x 2 / 2    26 x  26 x 512   ->    13 x  13 x 512
   18 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   19 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
   20 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   21 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
   22 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   23 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024
   24 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024
   25 route  16
   26 reorg              / 2    26 x  26 x 512   ->    13 x  13 x2048
   27 route  26 24
   28 conv   1024  3 x 3 / 1    13 x  13 x3072   ->    13 x  13 x1024
   29 conv     35  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x  35
   30 detection
Loading weights from input.weights...mj = 0, mn = 1, *(net->seen) = 19200
load_convolutional_weights: l.n*l.c*l.size*l.size = 864
load_convolutional_weights: l.n*l.c*l.size*l.size = 18432
load_convolutional_weights: l.n*l.c*l.size*l.size = 73728
load_convolutional_weights: l.n*l.c*l.size*l.size = 8192
load_convolutional_weights: l.n*l.c*l.size*l.size = 73728
load_convolutional_weights: l.n*l.c*l.size*l.size = 294912
load_convolutional_weights: l.n*l.c*l.size*l.size = 32768
load_convolutional_weights: l.n*l.c*l.size*l.size = 294912
load_convolutional_weights: l.n*l.c*l.size*l.size = 1179648
load_convolutional_weights: l.n*l.c*l.size*l.size = 131072
load_convolutional_weights: l.n*l.c*l.size*l.size = 1179648
load_convolutional_weights: l.n*l.c*l.size*l.size = 131072
load_convolutional_weights: l.n*l.c*l.size*l.size = 1179648
load_convolutional_weights: l.n*l.c*l.size*l.size = 4718592
load_convolutional_weights: l.n*l.c*l.size*l.size = 524288
load_convolutional_weights: l.n*l.c*l.size*l.size = 4718592
load_convolutional_weights: l.n*l.c*l.size*l.size = 524288
load_convolutional_weights: l.n*l.c*l.size*l.size = 4718592
load_convolutional_weights: l.n*l.c*l.size*l.size = 9437184
load_convolutional_weights: l.n*l.c*l.size*l.size = 9437184
load_convolutional_weights: l.n*l.c*l.size*l.size = 28311552
load_convolutional_weights: l.n*l.c*l.size*l.size = 35840
Done!
Image data = 0x1e729100, w = 992, h = 620
Detected 1 objects
Box #0: x,y,w,h = [0.406521, 0.283362, 0.383790, 0.508666]

Exiting...


NOTES:

- Cleanup in the destructor is not properly done (core issue in darknet itself). This will be resolved in a later release.
