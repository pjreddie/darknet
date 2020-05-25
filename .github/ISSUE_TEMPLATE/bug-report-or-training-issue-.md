---
name: 'Bug report or Training issue '
about: Create a report to help us improve
title: ''
labels: I think a bug here
assignees: ''

---

If you want to report a bug - provide:
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
