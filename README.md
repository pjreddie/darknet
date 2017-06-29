# Yolo-Windows v2

1. [How to use](#how-to-use)
2. [How to compile](#how-to-compile)
3. [How to train (Pascal VOC Data)](#how-to-train-pascal-voc-data)
4. [How to train (to detect your custom objects)](#how-to-train-to-detect-your-custom-objects)
5. [When should I stop training](#when-should-i-stop-training)
6. [How to improve object detection](#how-to-improve-object-detection)
7. [How to mark bounded boxes of objects and create annotation files](#how-to-mark-bounded-boxes-of-objects-and-create-annotation-files)
8. [How to use Yolo as DLL](#how-to-use-yolo-as-dll)

|  ![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png) | &nbsp; ![map_fps](https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg) https://arxiv.org/abs/1612.08242 |
|---|---|

|  ![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png) | &nbsp; ![map_fps](https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg) https://arxiv.org/abs/1612.08242 |
|---|---|


# "You Only Look Once: Unified, Real-Time Object Detection (version 2)"
A yolo windows version (for object detection)

Contributtors: https://github.com/pjreddie/darknet/graphs/contributors

This repository is forked from Linux-version: https://github.com/pjreddie/darknet

More details: http://pjreddie.com/darknet/yolo/

##### Requires: 
* **MS Visual Studio 2015 (v140)**: https://go.microsoft.com/fwlink/?LinkId=532606&clcid=0x409  (or offline [ISO image](https://go.microsoft.com/fwlink/?LinkId=615448&clcid=0x409))
* **CUDA 8.0 for Windows x64**: https://developer.nvidia.com/cuda-downloads
* **OpenCV 2.4.9**: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe/download
  - To compile without OpenCV - remove define OPENCV from: Visual Studio->Project->Properties->C/C++->Preprocessor
  - To compile with different OpenCV version - change in file yolo.c each string look like **#pragma comment(lib, "opencv_core249.lib")** from 249 to required version.
  - With OpenCV will show image or video detection in window and store result to: test_dnn_out.avi

##### Pre-trained models for different cfg-files can be downloaded from (smaller -> faster & lower quality):
* `yolo.cfg` (256 MB COCO-model) - require 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo.weights
* `yolo-voc.cfg` (256 MB VOC-model) - require 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo-voc.weights
* `tiny-yolo.cfg` (60 MB COCO-model) - require 1 GB GPU-RAM: http://pjreddie.com/media/files/tiny-yolo.weights
* `tiny-yolo-voc.cfg` (60 MB VOC-model) - require 1 GB GPU-RAM: http://pjreddie.com/media/files/tiny-yolo-voc.weights

Put it near compiled: darknet.exe

You can get cfg-files by path: `darknet/cfg/`

##### Examples of results:

[![Everything Is AWESOME](http://img.youtube.com/vi/VOC3huqHrss/0.jpg)](https://www.youtube.com/watch?v=VOC3huqHrss "Everything Is AWESOME")

Others: https://www.youtube.com/channel/UC7ev3hNVkx4DzZ3LO19oebg

### How to use:

##### Example of usage in cmd-files from `build\darknet\x64\`:

* `darknet_voc.cmd` - initialization with 256 MB VOC-model yolo-voc.weights & yolo-voc.cfg and waiting for entering the name of the image file
* `darknet_demo_voc.cmd` - initialization with 256 MB VOC-model yolo-voc.weights & yolo-voc.cfg and play your video file which you must rename to: test.mp4, and store result to: test_dnn_out.avi
* `darknet_net_cam_voc.cmd` - initialization with 256 MB VOC-model, play video from network video-camera mjpeg-stream (also from you phone) and store result to: test_dnn_out.avi
* `darknet_web_cam_voc.cmd` - initialization with 256 MB VOC-model, play video from Web-Camera number #0 and store result to: test_dnn_out.avi

##### How to use on the command line:
* 256 MB COCO-model - image: `darknet.exe detector test data/coco.data yolo.cfg yolo.weights -i 0 -thresh 0.2`
* Alternative method 256 MB COCO-model - image: `darknet.exe detect yolo.cfg yolo.weights -i 0 -thresh 0.2`
* 256 MB VOC-model - image: `darknet.exe detector test data/voc.data yolo-voc.cfg yolo-voc.weights -i 0`
* 256 MB COCO-model - video: `darknet.exe detector demo data/coco.data yolo.cfg yolo.weights test.mp4 -i 0`
* 256 MB VOC-model - video: `darknet.exe detector demo data/voc.data yolo-voc.cfg yolo-voc.weights test.mp4 -i 0`
* Alternative method 256 MB VOC-model - video: `darknet.exe yolo demo yolo-voc.cfg yolo-voc.weights test.mp4 -i 0`
* 60 MB VOC-model for video: `darknet.exe detector demo data/voc.data tiny-yolo-voc.cfg tiny-yolo-voc.weights test.mp4 -i 0`
* 256 MB COCO-model for net-videocam - Smart WebCam: `darknet.exe detector demo data/coco.data yolo.cfg yolo.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`
* 256 MB VOC-model for net-videocam - Smart WebCam: `darknet.exe detector demo data/voc.data yolo-voc.cfg yolo-voc.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`
* 256 MB VOC-model - WebCamera #0: `darknet.exe detector demo data/voc.data yolo-voc.cfg yolo-voc.weights -c 0`

##### For using network video-camera mjpeg-stream with any Android smartphone:

1. Download for Android phone mjpeg-stream soft: IP Webcam / Smart WebCam


    * Smart WebCam - preferably: https://play.google.com/store/apps/details?id=com.acontech.android.SmartWebCam2
    * IP Webcam: https://play.google.com/store/apps/details?id=com.pas.webcam

2. Connect your Android phone to computer by WiFi (through a WiFi-router) or USB
3. Start Smart WebCam on your phone
4. Replace the address below, on shown in the phone application (Smart WebCam) and launch:


* 256 MB COCO-model: `darknet.exe detector demo data/coco.data yolo.cfg yolo.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`
* 256 MB VOC-model: `darknet.exe detector demo data/voc.data yolo-voc.cfg yolo-voc.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`


### How to compile:

1. If you have MSVS 2015, CUDA 8.0 and OpenCV 2.4.9 (with paths: `C:\opencv_2.4.9\opencv\build\include` & `C:\opencv_2.4.9\opencv\build\x64\vc12\lib` or `vc14\lib`), then start MSVS, open `build\darknet\darknet.sln`, set **x64** and **Release**, and do the: Build -> Build darknet

  1.1. Find files `opencv_core249.dll`, `opencv_highgui249.dll` and `opencv_ffmpeg249_64.dll` in `C:\opencv_2.4.9\opencv\build\x64\vc12\bin` or `vc14\bin` and put it near with `darknet.exe`

2. If you have other version of CUDA (not 8.0) then open `build\darknet\darknet.vcxproj` by using Notepad, find 2 places with "CUDA 8.0" and change it to your CUDA-version, then do step 1

3. If you have other version of OpenCV 2.4.x (not 2.4.9) then you should change pathes after `\darknet.sln` is opened

  3.1 (right click on project) -> properties  -> C/C++ -> General -> Additional Include Directories
  
  3.2 (right click on project) -> properties  -> Linker -> General -> Additional Library Directories
  
  3.3 Open file: `\src\yolo.c` and change 3 lines to your OpenCV-version - `249` (for 2.4.9), `2413` (for 2.4.13), ... : 

    * `#pragma comment(lib, "opencv_core249.lib")`
    * `#pragma comment(lib, "opencv_imgproc249.lib")`
    * `#pragma comment(lib, "opencv_highgui249.lib")` 


4. If you have other version of OpenCV 3.x (not 2.4.x) then you should change many places in code by yourself.

5. If you want to build with CUDNN to speed up then:
      
    * download and install **cuDNN 5.1 for CUDA 8.0**: https://developer.nvidia.com/cudnn
      
    * add Windows system variable `cudnn` with path to CUDNN: https://hsto.org/files/a49/3dc/fc4/a493dcfc4bd34a1295fd15e0e2e01f26.jpg
      
    * open `\darknet.sln` -> (right click on project) -> properties  -> C/C++ -> Preprocessor -> Preprocessor Definitions, and add at the beginning of line: `CUDNN;`

### How to compile (custom):

Also, you can to create your own `darknet.sln` & `darknet.vcxproj`, this example for CUDA 8.0 and OpenCV 2.4.9

Then add to your created project:
- (right click on project) -> properties  -> C/C++ -> General -> Additional Include Directories, put here: 

`C:\opencv_2.4.9\opencv\build\include;..\..\3rdparty\include;%(AdditionalIncludeDirectories);$(CudaToolkitIncludeDir);$(cudnn)\include`
- (right click on project) -> Build dependecies -> Build Customizations -> set check on CUDA 8.0 or what version you have - for example as here: http://devblogs.nvidia.com/parallelforall/wp-content/uploads/2015/01/VS2013-R-5.jpg
- add to project all .c & .cu files from `\src`
- (right click on project) -> properties  -> Linker -> General -> Additional Library Directories, put here: 

`C:\opencv_2.4.9\opencv\build\x64\vc12\lib;$(CUDA_PATH)lib\$(PlatformName);$(cudnn)\lib\x64;%(AdditionalLibraryDirectories)`
-  (right click on project) -> properties  -> Linker -> Input -> Additional dependecies, put here: 

`..\..\3rdparty\lib\x64\pthreadVC2.lib;cublas.lib;curand.lib;cudart.lib;cudnn.lib;%(AdditionalDependencies)`
- (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions

`OPENCV;_TIMESPEC_DEFINED;_CRT_SECURE_NO_WARNINGS;GPU;WIN32;NDEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)`

- open file: `\src\yolo.c` and change 3 lines to your OpenCV-version - `249` (for 2.4.9), `2413` (for 2.4.13), ... : 

    * `#pragma comment(lib, "opencv_core249.lib")`
    * `#pragma comment(lib, "opencv_imgproc249.lib")`
    * `#pragma comment(lib, "opencv_highgui249.lib")` 

- compile to .exe (X64 & Release) and put .dll-s near with .exe:

`pthreadVC2.dll, pthreadGC2.dll` from \3rdparty\dll\x64

`cusolver64_80.dll, curand64_80.dll, cudart64_80.dll, cublas64_80.dll` - 80 for CUDA 8.0 or your version, from C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin

`opencv_core249.dll`, `opencv_highgui249.dll` and `opencv_ffmpeg249_64.dll` in `C:\opencv_2.4.9\opencv\build\x64\vc12\bin` or `vc14\bin`

## How to train (Pascal VOC Data):

1. Download pre-trained weights for the convolutional layers (76 MB): http://pjreddie.com/media/files/darknet19_448.conv.23 and put to the directory `build\darknet\x64`

2. Download The Pascal VOC Data and unpack it to directory `build\darknet\x64\data\voc` will be created dir `build\darknet\x64\data\voc\VOCdevkit\`:
    * http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    * http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    * http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    
    2.1 Download file `voc_label.py` to dir `build\darknet\x64\data\voc`: http://pjreddie.com/media/files/voc_label.py

3. Download and install Python for Windows: https://www.python.org/ftp/python/3.5.2/python-3.5.2-amd64.exe

4. Run command: `python build\darknet\x64\data\voc\voc_label.py` (to generate files: 2007_test.txt, 2007_train.txt, 2007_val.txt, 2012_train.txt, 2012_val.txt)

5. Run command: `type 2007_train.txt 2007_val.txt 2012_*.txt > train.txt`

6. Set `batch=64` and `subdivisions=8` in the file `yolo-voc.2.0.cfg`: [link](https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/yolo-voc.cfg#L3)

7. Start training by using `train_voc.cmd` or by using the command line: `darknet.exe detector train data/voc.data yolo-voc.2.0.cfg darknet19_448.conv.23`

If required change pathes in the file `build\darknet\x64\data\voc.data`

More information about training by the link: http://pjreddie.com/darknet/yolo/#train-voc

## How to train with multi-GPU:

1. Train it first on 1 GPU for like 1000 iterations: `darknet.exe detector train data/voc.data yolo-voc.2.0.cfg darknet19_448.conv.23`

2. Then stop and by using partially-trained model `/backup/yolo-voc_1000.weights` run training with multigpu (up to 4 GPUs): `darknet.exe detector train data/voc.data yolo-voc.2.0.cfg yolo-voc_1000.weights -gpus 0,1,2,3`

https://groups.google.com/d/msg/darknet/NbJqonJBTSY/Te5PfIpuCAAJ

## How to train (to detect your custom objects):

1. Create file `yolo-obj.cfg` with the same content as in `yolo-voc.2.0.cfg` (or copy `yolo-voc.2.0.cfg` to `yolo-obj.cfg)` and:

  * change line batch to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/yolo-voc.cfg#L3)
  * change line subdivisions to [`subdivisions=8`](https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/yolo-voc.cfg#L4)
  * change line `classes=20` to your number of objects
  * change line #237 from [`filters=125`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolo-voc.cfg#L237) to `filters=(classes + 5)*5` (generally this depends on the `num` and `coords`, i.e. equal to `(classes + coords + 1)*num`)

  For example, for 2 objects, your file `yolo-obj.cfg` should differ from `yolo-voc.2.0.cfg` in such lines:

  ```
  [convolutional]
  filters=35

  [region]
  classes=2
  ```

2. Create file `obj.names` in the directory `build\darknet\x64\data\`, with objects names - each in new line

3. Create file `obj.data` in the directory `build\darknet\x64\data\`, containing (where **classes = number of objects**):

  ```
  classes= 2
  train  = data/train.txt
  valid  = data/test.txt
  names = data/obj.names
  backup = backup/
  ```

4. Put image-files (.jpg) of your objects in the directory `build\darknet\x64\data\obj\`

5. Create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line: `<object-class> <x> <y> <width> <height>`

  Where: 
  * `<object-class>` - integer number of object from `0` to `(classes-1)`
  * `<x> <y> <width> <height>` - float values relative to width and height of image, it can be equal from 0.0 to 1.0 
  * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
  * atention: `<x> <y>` - are center of rectangle (are not top-left corner)

  For example for `img1.jpg` you should create `img1.txt` containing:

  ```
  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667
  ```

6. Create file `train.txt` in directory `build\darknet\x64\data\`, with filenames of your images, each filename in new line, with path relative to `darknet.exe`, for example containing:

  ```
  data/obj/img1.jpg
  data/obj/img2.jpg
  data/obj/img3.jpg
  ```

7. Download pre-trained weights for the convolutional layers (76 MB): http://pjreddie.com/media/files/darknet19_448.conv.23 and put to the directory `build\darknet\x64`

8. Start training by using the command line: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet19_448.conv.23`

    (file `yolo-obj_xxx.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations until 1000 iterations has been reached, and after for each 1000 iterations)

9. After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

 * After each 1000 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just copy `yolo-obj_2000.weights` from `build\darknet\x64\backup\` to `build\darknet\x64\` and start training using: `darknet.exe detector train data/obj.data yolo-obj.cfg yolo-obj_2000.weights`

 * Also you can get result earlier than all 45000 iterations.
 
## When should I stop training:

Usually sufficient 2000 iterations for each class(object). But for a more precise definition when you should stop training, use the following manual:

1. During training, you will see varying indicators of error, and you should stop when no longer decreases **0.060730 avg**:

  > Region Avg IOU: 0.798363, Class: 0.893232, Obj: 0.700808, No Obj: 0.004567, Avg Recall: 1.000000,  count: 8
  > Region Avg IOU: 0.800677, Class: 0.892181, Obj: 0.701590, No Obj: 0.004574, Avg Recall: 1.000000,  count: 8
  >
  > **9002**: 0.211667, **0.060730 avg**, 0.001000 rate, 3.868000 seconds, 576128 images
  > Loaded: 0.000000 seconds

  * **9002** - iteration number (number of batch)
  * **0.060730 avg** - average loss (error) - **the lower, the better**

  When you see that average loss **0.xxxxxx avg** no longer decreases at many iterations then you should stop training.

2. Once training is stopped, you should take some of last `.weights`-files from `darknet\build\darknet\x64\backup` and choose the best of them:

For example, you stopped training after 9000 iterations, but the best result can give one of previous weights (7000, 8000, 9000). It can happen due to overfitting. **Overfitting** - is case when you can detect objects on images from training-dataset, but can't detect ojbects on any others images. You should get weights from **Early Stopping Point**:

![Overfitting](https://hsto.org/files/5dc/7ae/7fa/5dc7ae7fad9d4e3eb3a484c58bfc1ff5.png) 

To get weights from Early Stopping Point:

  2.1. At first, in your file `obj.data` you must specify the path to the validation dataset `valid = valid.txt` (format of `valid.txt` as in `train.txt`), and if you haven't validation images, just copy `data\train.txt` to `data\valid.txt`.

  2.2 If training is stopped after 9000 iterations, to validate some of previous weights use this commands:

* `darknet.exe detector recall data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
* `darknet.exe detector recall data/obj.data yolo-obj.cfg backup\yolo-obj_8000.weights`
* `darknet.exe detector recall data/obj.data yolo-obj.cfg backup\yolo-obj_9000.weights`

And comapre last output lines for each weights (7000, 8000, 9000):

> 7586 7612 7689 RPs/Img: 68.23 **IOU: 77.86%** Recall:99.00%

* **IOU** - the bigger, the better (says about accuracy) - **better to use**
* **Recall** - the bigger, the better (says about accuracy) - actually Yolo calculates true positives, so it shouldn't be used

For example, **bigger IOU** gives weights `yolo-obj_8000.weights` - then **use this weights for detection**.


![precision_recall_iou](https://hsto.org/files/ca8/866/d76/ca8866d76fb840228940dbf442a7f06a.jpg)

### Custom object detection:

Example of custom object detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`

| ![Yolo_v2_training](https://hsto.org/files/d12/1e7/515/d121e7515f6a4eb694913f10de5f2b61.jpg) | ![Yolo_v2_training](https://hsto.org/files/727/c7e/5e9/727c7e5e99bf4d4aa34027bb6a5e4bab.jpg) |
|---|---|

## How to improve object detection:

1. Before training:
  * set flag `random=1` in your `.cfg`-file - it will increase precision by training Yolo for different resolutions: [link](https://github.com/AlexeyAB/darknet/blob/47409529d0eb935fa7bafbe2b3484431117269f5/cfg/yolo-voc.cfg#L244)
  
  * desirable that your training dataset include images with objects at diffrent: scales, rotations, lightings, from different sides

2. After training - for detection:

  * Increase network-resolution by set in your `.cfg`-file (`height=608` and `width=608`) or (`height=832` and `width=832`) or (any value multiple of 32) - this increases the precision and makes it possible to detect small objects: [link](https://github.com/AlexeyAB/darknet/blob/47409529d0eb935fa7bafbe2b3484431117269f5/cfg/yolo-voc.cfg#L4)
  
    * you do not need to train the network again, just use `.weights`-file already trained for 416x416 resolution
    * if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64: [link](https://github.com/AlexeyAB/darknet/blob/47409529d0eb935fa7bafbe2b3484431117269f5/cfg/yolo-voc.cfg#L3)

## How to mark bounded boxes of objects and create annotation files:

Here you can find repository with GUI-software for marking bounded boxes of objects and generating annotation files for Yolo v2: https://github.com/AlexeyAB/Yolo_mark

With example of: `train.txt`, `obj.names`, `obj.data`, `yolo-obj.cfg`, `air`1-6`.txt`, `bird`1-4`.txt` for 2 classes of objects (air, bird) and `train_obj.cmd` with example how to train this image-set with Yolo v2

## How to use Yolo as DLL

1. To compile Yolo as C++ DLL-file `yolo_cpp_dll.dll` - open in MSVS2015 file `build\darknet\yolo_cpp_dll.sln`, set **x64** and **Release**, and do the: Build -> Build yolo_cpp_dll
    * You should have installed **CUDA 8.0**
    * To use cuDNN do: (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, and add at the beginning of line: `CUDNN;`

2. To use Yolo as DLL-file in your C++ console application - open in MSVS2015 file `build\darknet\yolo_console_dll.sln`, set **x64** and **Release**, and do the: Build -> Build yolo_console_dll

    * you can run your console application from Windows Explorer `build\darknet\x64\yolo_console_dll.exe`
    * or you can run from MSVS2015 (before this - you should copy 2 files `yolo-voc.cfg` and `yolo-voc.weights` to the directory `build\darknet\` )
    * after launching your console application and entering the image file name - you will see info for each object: 
    `<obj_id> <left_x> <top_y> <width> <height> <probability>`
    * to use simple OpenCV-GUI you should uncomment line `//#define OPENCV` in `yolo_console_dll.cpp`-file: [link](https://github.com/AlexeyAB/darknet/blob/a6cbaeecde40f91ddc3ea09aa26a03ab5bbf8ba8/src/yolo_console_dll.cpp#L5)
   
`yolo_cpp_dll.dll`-API: [link](https://github.com/AlexeyAB/darknet/blob/master/src/yolo_v2_class.hpp#L31)
```
class Detector {
public:
	Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
	~Detector();

	std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2);
	std::vector<bbox_t> detect(image_t img, float thresh = 0.2);

#ifdef OPENCV
	std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2);
#endif
};
```
