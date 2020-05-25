---
name: 'Bug report or Training issue '
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

Read the recommendations below if you want to make a Bug report, ask a Training question or request a Feature.


1. If you want to report a bug - provide:
    * description of a bug
    * what command do you use?
    * do you use Win/Linux/Mac?
    * attach screenshot of a bug with previous messages in terminal
    * in what cases a bug occurs, and in which not?
    * if possible, specify date/commit of Darknet that works without this bug
    * show such screenshot with info
```
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
 CUDA-version: 10000 (10000), cuDNN: 7.4.2, CUDNN_HALF=1, GPU count: 1
 CUDNN_HALF=1
 OpenCV version: 4.2.0
 0 : compute_capability = 750, cudnn_half = 1, GPU: GeForce RTX 2070
net.optimized_memory = 0
mini_batch = 1, batch = 8, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
```

----

2. If you have an issue with training - no-detections / Nan avg-loss / low accuracy:
    * what command do you use?
    * what dataset do you use?   
    * what Loss and mAP did you get?
    * show chart.png with Loss and mAP    
    * check your dataset - run training with flag `-show_imgs` i.e. `./darknet detector train ... -show_imgs` and look at the `aug_...jpg` images, do you see correct truth bounded boxes?
    * rename your cfg-file to txt-file and drag-n-drop (attach) to your message here
    * show content of generated files `bad.list` and `bad_label.list` if they exist
    * Read `How to train (to detect your custom objects)` and `How to improve object detection` in the Readme: https://github.com/AlexeyAB/darknet/blob/master/README.md
    * show such screenshot with info
```
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
 CUDA-version: 10000 (10000), cuDNN: 7.4.2, CUDNN_HALF=1, GPU count: 1
 CUDNN_HALF=1
 OpenCV version: 4.2.0
 0 : compute_capability = 750, cudnn_half = 1, GPU: GeForce RTX 2070
net.optimized_memory = 0
mini_batch = 1, batch = 8, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
```

----

3. For Feature-request:
    * describe your feature as detailed as possible
    * provide link to the paper and/or source code if it exist
    * attach chart/table with comparison that shows improvement
