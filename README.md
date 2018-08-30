![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Quick run the pretrained weights
---------------------
* Run on one Image
```
./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights ./data/dog.jpg
```
* Run on a series of images
```
./darknet detector list cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights /media/elab/sdd/data/TinyTLP/Sam/img/
```
* Reat-Time Detection on a Webcam
```
./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights
```

# Run with [OpenTracker](https://github.com/rockkingjy/OpenTracker)
-----------------
Copy trackers' source file from [[OpenTracker](https://github.com/rockkingjy/OpenTracker)] and make install:
```
git clone https://github.com/rockkingjy/OpenTracker.git
cd OpenTracker
make -j`nproc`
sudo make install
```
set flag in `makefile`:
```
OPENTRACKER=1
```
and run:
```
./darknet detector tracking cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights
```
It will first run the yolo detection. Then when a `person` comes into the vision field, then it will automatically do the tracking.

# Run with a touchscreen and OpenTracker:
---------------------
In `makefile` set:
```
TS=1
```
Assuming that your touchscreen is connected to /dev/input/event6, else modify it (must be run under sudo):
```
sudo ./darknet detector tracking cfg/coco.data cfg/yolov2-tiny.cfg yolov2-tiny.weights "/dev/input/event6"
```
or
```
sudo ./darknet detector tracking cfg/coco.data cfg/yolov3.cfg yolov3.weights "/dev/input/event4"
```
It will first run the yolo detection, with touchscreen, you choose the object to track, then it will track automatically.


# Run with maestro motor
--------------------
In Makefile set:
```
MAESTRO=1
```

# Train Imagenet
-----------------

Download the data and create the list file and put them in the data/ folder:
```
find `pwd`/ILSVRC2012_img_train -name \*.JPEG > imagenet1k.train.list
find `pwd`/ILSVRC2012_img_val -name \*.JPEG > imagenet1k.valid.list
```

# Training coco and voc
------------------
Remember to create backup_** before training
```
./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 1
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 -gpus 0
```

# Run on Mac
------------------
```
brew install opencv@2
echo 'export PATH="/usr/local/opt/opencv@2/bin:$PATH"' >> ~/.bash_profile
```
each time to open a new terminal, you should run these to make:
```
export LDFLAGS=-L/usr/local/opt/opencv@2/lib
export CPPFLAGS=-I/usr/local/opt/opencv@2/include
export PKG_CONFIG_PATH=/usr/local/opt/opencv@2/lib/pkgconfig
```

# Net compression (not finished)
-----------------
```
./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg
```
GPU=0 CUDNN=0 OPENMP=0: 21.315930 seconds.

GPU=0 CUDNN=0 OPENMP=1: 4.199370 seconds.

yolov3.cfg: width=208 height=208: 2.430844 seconds.


