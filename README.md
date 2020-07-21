![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

Yolo v4 paper: https://arxiv.org/abs/2004.10934

Yolo v4 source code: https://github.com/AlexeyAB/darknet

Yolov v4 tiny discussion: https://www.reddit.com/r/MachineLearning/comments/hu7lyt/p_yolov4tiny_speed_1770_fps_tensorrtbatch4/

Useful links: https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

# Usage
A detailed tutorial can be found [here](http://pjreddie.com/darknet/yolov2).

## Makefile
By default, the `GPU`, `CUDNN` and `OPENCV` settings in the `Makefile` are set to `0`.

In order to activate GPU and OpenCV support, change your `Makefile` settings as follows:
```
GPU=1
CUDNN=1
OPENCV=1
...
```

Then in the `darknet` repo folder, run `make` (or `make clean && make` if you compiled the project before).

## Detection (Image)
```
./darknet detect <config-file> <weights-file> <image-to-process> [-thresh <threshold>]
```

For example, in the darknet directory:
```
./darknet detect cfg/yolov2.cfg yolov2.weights data/dog.jpg
```

Alternative command:
```
./darknet detector test ...
```

Change paths according to description.


The darknet program will then process the image and output class probabilities. If your program was built with the OpenCV setting as `1`, it will also generate a `predictions.png` file with bounding boxes.


The default threshold is 0.5 (50%), but you can also specify the threshold:
```
./darknet detector test ... -thresh 0.3
```

## Detection (Video)
Darknet also has a demo video detector, but you will need to compile Darknet with CUDA and OpenCV. Then run the command:

```
./darknet detector demo <path_to_data_file> <path_to_config_file> <path_to_weights_file> <path_to_video_file (optional: default is webcam)>
```

For example to run on webcam,
```
./darknet detector demo cfg/coco.data cfg/yolov2.cfg yolov2.weights
```

## Training
Follow the instructions [here](https://pjreddie.com/darknet/yolov2/) for training data on [VOC](https://pjreddie.com/darknet/yolov2/#train-voc) and [COCO](https://pjreddie.com/darknet/yolov2).
