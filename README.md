![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

# Run on a series of images
```
./darknet detector list cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights /media/elab/sdd/data/TinyTLP/Sam/img/
```
# Run on one Image
```
./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights /media/elab/sdd/data/TLP/Sam/img/00001.jpg
```
# Reat-Time Detection on a Webcam
```
./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights
```
# Training
Remember to create backup_** before training
```
./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 1
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 -gpus 0
```
# Net compression
```
./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg
```
GPU=0 CUDNN=0 OPENMP=0: 21.315930 seconds.

GPU=0 CUDNN=0 OPENMP=1: 4.199370 seconds.

yolov3.cfg: width=208 height=208: 2.430844 seconds.


