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

		bool GetBoxes(box* outBoxes, std::string* outLabels, int boxCount);
		

Steps to build arapaho on Linux:

1. Build the darknet shared library

$ cd darknet
$ make darknet-cpp-shared

This will generate the shared library, libdarknet-cpp-shared.so. Copy this into arapaho folder

2. Copy the weights, cfg, data and image files for detection into the arapaho folder. These should be named as input.cfg, input.data, input.weights, input.jpg or input.mp4 etc

$ cd arapaho
$ ls input*
input.cfg  input.data  input.jpg  input.weights

3. Build arapaho test app (including the arapaho core) for detection, using below command

make arapaho

NOTE: GPU and CUDNN flags have to match what was specified in the build of darknet library, in step 1. Change in the arapaho Makefile appropriately.

4. Run arapaho test wrapper as below

$ ./arapaho.out

This will generate the below output from the test application. Using GetBoxes() API, this can be easily integrated into any C++ application

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
Image data = 0x1de6e2c0, w = 256, h = 145
Detect: Resizing image to match network 
l.softmax_tree = (nil), nms = 0.400000
==> Detected [1] objects in [0.160705] seconds
Box #0: center {x,y}, box {w,h} = [0.173209, 0.341205, 0.252983, 0.250536]
Label:<label_name>
Exiting...


NOTES:

- Cleanup in the destructor is not properly done (core issue in darknet itself). This will be resolved in a later release.

- For memory usage on GPU, refer to https://github.com/prabindh/darknet/blob/master/arapaho/Arapaho-GPU-Z%20Sensor-Log.txt

- Windows build of Arapaho is supported through .sln files at https://github.com/prabindh/darknet-cpp-windows