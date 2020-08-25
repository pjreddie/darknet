---
name: Training issue - no-detections / Nan avg-loss / low accuracy
about: Training issue - no-detections / Nan avg-loss / low accuracy
title: ''
labels: Training issue
assignees: ''

---

If you have an issue with training - no-detections / Nan avg-loss / low accuracy:
    * read FAQ: https://github.com/AlexeyAB/darknet/wiki/FAQ---frequently-asked-questions
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
