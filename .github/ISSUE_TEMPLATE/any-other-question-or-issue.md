---
name: Any other question or issue
about: Any other question or issue
title: ''
labels: ''
assignees: ''

---

If something doesn’t work for you, then show 2 screenshots:
1. screenshots of your issue
2. screenshots with such information
```
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
 CUDA-version: 10000 (10000), cuDNN: 7.4.2, CUDNN_HALF=1, GPU count: 1
 CUDNN_HALF=1
 OpenCV version: 4.2.0
 0 : compute_capability = 750, cudnn_half = 1, GPU: GeForce RTX 2070
net.optimized_memory = 0
mini_batch = 1, batch = 8, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
   0 conv     32       3 x 3/ 1    608 x 608 x   3 ->  608 x 608 x  32 0.639 BF
```

If you do not get an answer for a long time, try to find the answer among Issues with a Solved label: https://github.com/AlexeyAB/darknet/issues?q=is%3Aopen+is%3Aissue+label%3ASolved
