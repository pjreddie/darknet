## Compilation  编译

```c
// example/detection.c
// :138
    // 这里是保存weights方式，超过1000步，就10000步保存一次；少于1000步就100步保存一次
    if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
        if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
        char buff[256];
        sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
        save_weights(net, buff);
    }
```

也可以改变为 `if(i%1000==0 || (i < 1000 && i%100 == 0)){`  每1000步保存一次

```shell
sh$ vi Makefile
|[
GPU=1     # with CUDA to accelerate by using GPU (CUDA should be in /usr/local/cuda)
CUDNN=1   # with cuDNN v5-v7 to accelerate training by using GPU (should be in /usr/local/cudnn)
OPENCV=0  # to detect on video
OPENMP=1  # to support using multi-core CPU
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
#  用于BGD(Batch Gradient Descent) 批梯度下降算法，
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
# 
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
# 
steps=10000,25000
# 学习率变化比率，和 steps 个数一致
# after 10000, multiply the learning_rate by 0.1, then after 25000 multiply again by 0.1
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
# 激活函数，logistic, loggy, relu, elu, relie, plse, hardtan, lhtan, linear, ramp, leaky, tanh, stair
activation=leaky



[yolo]
mask = 6,7,8
# 预测框初始宽高
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=9
num=9
# 抖动增加噪声来抑制过拟合
jitter=.3
ignore_thresh = .5
truth_thresh = 1
# increase precision by training Yolo for differenct resolutions
# 是否启用Multi-Scale Training，随机使用不同尺寸图片进行训练；若为0，训练大小跟输入大小一致
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
train  = 3d_data/train.txt           ; 这里包含训练图片集，并包含一个.txt 标注数据； 如 /tmp/a.jpg  ==> 表示同时包含 /tmp/a.jpg 和 annotation/tmp/a.txt
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
# @param -thresh default to 0.25, displays objects detected with a confidence of .25+
sh$ ./darknet detect cfg/yolov3.cfg trained.weights data/dog.jpg -thresh 0.25

sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights data/dog.jpg -thresh 0.25
sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights -thresh 0.25
sh$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg trained.weights -thresh 0.25
```

### When should I stop training

1. No longer decreases 0.xxxx avg        --->  average loss error, the lower, the better
2. Get weights from **Early Stopping Point**

* **IoU** (Intersect of Union)
* **mAP** (Mean Average Precision)

### How to improve object detection

1. Before training
  * set flag `random=1` and increase `width` and `height` in `.cfg`
  * recalculate anchors, and set the anchors to `yolo.anchors` in `.cfg`
    * `sh$ ./darknet detector calc_anchors data/xx.data -num_of_clusters 9 -width 416 -height 416`

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

* `Region Avg IOU: 0.702134` the average of the IoU of every image in the current *subdivision*. A 70.2134% overlap in this case. 代表预测的bounding box和ground truth的交集与并集之比，期望该值趋近于1。
* `Class: 0.942936` 是标注物体的概率，期望该值趋近于1.
* `Obj: 0.804691` 期望该值趋近于1.
* `No Obj: 0.015181` 期望该值越来越小但不为零.
* `Avg Recall` is defined the code as `recall/count`, and thus a metric for how many *positives* detected out of the total amount of positives in this subdivision. 期望该值趋近1
* `count: 82` the amount of *positives* (objects to be detected).