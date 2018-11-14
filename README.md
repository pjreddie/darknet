# Darknet

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Compilation


```shell
sh$ vi Makefile
|[
GPU=1     # default is 0, with CUDA to accelerate by using GPU (CUDA should be in /usr/local/cuda)
CUDNN=1   # default is 0, with cuDNN v5-v7 to accelerate training by using GPU (should be in /usr/local/cudnn)
OPENCV=0  # to detect on video
OPENMP=1  # default is 0, to support using multi-core CPU
DEBUG=0
]|

sh$ vi ~/.bash_profile
|[
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
]|
sh$ make clean
sh$ make
```

* Coords: x, y, width, height

## cfg

### .cfg

 ```conf
[net]
#  for BGD(Batch Gradient Descent)
batch=32
# divide 1 batch into N sub-batches
subdivisions=16
# width/height must be divisible by 32, increase it will increase precision
# it may cause `out of memory (GPU)`
width=640
height=640
# RGB
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

# learning_rate * GPUs = 0.001; 4GPUs = 0.00025; 8GPUs=0.000125
learning_rate=0.000125
# burn_in = GPUs * 1000
#  if(batch_num < net.burn_in) {
#      return net.learning_rate * pow((float)batch_num / net.burn_in, net.power)
#  }
burn_in=8000
# max_batches = GPUs * 6000
max_batches = 48000
# constant, step exp, poly, steps, sig, random
policy=steps
steps=10000,25000
# after 10000, multiply the learning_rate by 0.1, then after 25000 multiply again by 0.1
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
# activation function: ogistic, loggy, relu, elu, relie, plse, hardtan, lhtan, linear, ramp, leaky, tanh, stair
activation=leaky

[yolo]
mask = 6,7,8
# initial width and height of prediction boxes
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=9
num=9
# increase jitter to avoid overfitting
jitter=.3
ignore_thresh = .5
truth_thresh = 1
# increase precision by training Yolo for differenct resolutions
random=1


[convolutional]
# filters = (next [yolo].classes+coords+1)*numberOfMask = (20+5)\*3
filters=75
[yolo]
classes=20


[convolutional]
filters=45

[yolo]
classes=10
```

### .data

```ini
classes= 4
; training images
; @notice there must be one annotation data (e.g. /tmp/a.txt) for each training image (/tmp/a.jpg).
train  = 3d_data/train.txt
valid  = 3d_data/test.txt
names = data/3d_4sku.names
backup = backup/3d_4sku_2_6000
```

## Commands

```shell
# Train the model
sh$ ./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74
sh$ ./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 0,1,2,3

# Stop and restart training from a checkpoint
sh$ ./darknet detector train cfg/coco.data cfg/yolov3.cfg backup/yolov3.backup

# Test
# @notice set batch=1 and subdivisions=1 when do test
#   `detect` is shorthand for `detector test cfg/coco.data`
# @param -thresh default to 0.25, displays objects detected with a confidence of .5+
# @out existed directory or file path; output the prediction(s) into this directory or this file path.
sh$ ./darknet detect cfg/yolov3.cfg trained.weights data/dog.jpg -thresh 0.5

sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights data/valid.txt -out predictions.jpg -thresh 0.5 -gpus 0
sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights data/valid.txt -out /tmp -thresh 0.5
sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights data/dog.jpg -thresh 0.5
sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights -thresh 0.5
sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights -thresh 0.5
```

### When should I stop training

1. No longer decreases 0.xxxx avg        --->  average loss error, the lower, the better
2. Get weights from **Early Stopping Point**

* **IoU** (Intersect of Union)
* **mAP** (Mean Average Precision)

### How to improve object detection

> Before training
>> set flag `random=1` and increase `width` and `height` in `.cfg`
>> recalculate anchors, and set the anchors to `yolo.anchors` in `.cfg`
>>> `sh$ ./darknet detector calc_anchors data/xx.data -num_of_clusters 9 -width 416 -height 416`

## Train Log

Let's have a look at IOU (Intersection over Union, also known as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index).

![IoU](https://timebutt.github.io/static/content/images/2017/06/Intersection_over_Union_-_visual_equation-1.png)

```cfg
[net]
batch=64
subdivisions=16
```

The output below is generated in detector.c on [line 136](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/detector.c#L136).

```log
...
Region 94 Avg IOU: 0.702134, Class: 0.942936, Obj: 0.804691, No Obj: 0.015181, .5R: 0.963415, .75R: 0.341463,  count: 82
Region 106 Avg IOU: 0.755752, Class: 0.995655, Obj: 0.898871, No Obj: 0.003263, .5R: 0.904762, .75R: 0.666667,  count: 63
6211: 4.289178, 5.320355 avg, 0.000363 rate, 11.845679 seconds, 4770048 images
Loaded: 0.000145 seconds
Region 82 Avg IOU: -nan, Class: -nan, Obj: -nan, No Obj: 0.000060, .5R: -nan, .75R: -nan,  count: 0
```

** A Batch Result

* `6211` the current training iteration/batch
* `4.28917`8` total loss
* `5.320355 avg` average loss error. **As a rule of thumb, once this reaches below 0.060730 avg, you can stop training.**
* `0.000363 rate` current learning rate, as defined in the `.cfg`
* `11.845679 seconds` time spent to process this batch
* `4770048 images` should less than 6211 * batch, the total amount of images used during training so far.

** Subdivision Output

* `Region Avg IOU: 0.702134` the average of the IoU of every image in the current *subdivision*. A 70.2134% overlap in this case. 
* `Class: 0.942936`
* `Obj: 0.804691`
* `No Obj: 0.015181`
* `Avg Recall` is defined the code as `recall/count`, and thus a metric for how many *positives* detected out of the total amount of positives in this subdivision.
* `count: 82` the amount of *positives* (objects to be detected).