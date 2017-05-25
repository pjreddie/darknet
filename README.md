# Darknet
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

## Results - 4K video

### YOLO COCO

[![4K YOLO COCO Object Detection #1](http://img.youtube.com/vi/yQwfDxBMtXg/0.jpg)](https://youtu.be/yQwfDxBMtXg?list=PLki8NgAAjqCb1NoCqpeZJHDzoy6y1Rx_H)

### YOLO VOC

[![4K YOLO COCO Object Detection #1](http://img.youtube.com/vi/yCOwvJw_7EI/0.jpg)](https://youtu.be/yCOwvJw_7EI?list=PLki8NgAAjqCb1NoCqpeZJHDzoy6y1Rx_H)


### Tiny YOLO VOC

...

### YOLO 9000

...

## Webcam Demo!

Try it yourself:
1. Build darknet
2. Download weights
3. Run webcam demos:
```
./webcam-coco.sh
./webcam-tiny-yolo.sh
./webcam-voc.sh
./webcam-yolo9000.sh
```

## Build darknet

Edit `Makefile` to enable GPU, CUDNN and OpenCV:

```
GPU=1
CUDNN=1
OPENCV=1
DEBUG=0
```

Choose your CUDA architecture, example for GTX980M: (you can check it here [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus))

```
ARCH=       -gencode arch=compute_52,code=[sm_52,compute_52]
```

Run `make` and you are ready!

## Download weights

All weights are available at [Darknet project website](http://pjreddie.com/darknet).

```
cd weights
wget https://pjreddie.com/media/files/yolo-voc.weights
wget -O yolo-coco.weights https://pjreddie.com/media/files/yolo.weights
wget https://pjreddie.com/media/files/tiny-yolo-voc.weights
wget https://pjreddie.com/media/files/yolo9000.weights
```

## Architectures

### YOLO COCO

|      | layer|filters|    size   |       input      |     output      |
|------|------|-------|-----------|------------------|-----------------|
|    0 | conv |    32 | 3 x 3 / 1 |  608 x 608 x   3 | 608 x 608 x  32 |
|    1 | max  |       | 2 x 2 / 2 |  608 x 608 x  32 | 304 x 304 x  32 |
|    2 | conv |    64 | 3 x 3 / 1 |  304 x 304 x  32 | 304 x 304 x  64 |
|    3 | max  |       | 2 x 2 / 2 |  304 x 304 x  64 | 152 x 152 x  64 |
|    4 | conv |   128 | 3 x 3 / 1 |  152 x 152 x  64 | 152 x 152 x 128 |
|    5 | conv |    64 | 1 x 1 / 1 |  152 x 152 x 128 | 152 x 152 x  64 |
|    6 | conv |   128 | 3 x 3 / 1 |  152 x 152 x  64 | 152 x 152 x 128 |
|    7 | max  |       | 2 x 2 / 2 |  152 x 152 x 128 |  76 x  76 x 128 |
|    8 | conv |   256 | 3 x 3 / 1 |   76 x  76 x 128 |  76 x  76 x 256 |
|    9 | conv |   128 | 1 x 1 / 1 |   76 x  76 x 256 |  76 x  76 x 128 |
|   10 | conv |   256 | 3 x 3 / 1 |   76 x  76 x 128 |  76 x  76 x 256 |
|   11 | max  |       | 2 x 2 / 2 |   76 x  76 x 256 |  38 x  38 x 256 |
|   12 | conv |   512 | 3 x 3 / 1 |   38 x  38 x 256 |  38 x  38 x 512 |
|   13 | conv |   256 | 1 x 1 / 1 |   38 x  38 x 512 |  38 x  38 x 256 |
|   14 | conv |   512 | 3 x 3 / 1 |   38 x  38 x 256 |  38 x  38 x 512 |
|   15 | conv |   256 | 1 x 1 / 1 |   38 x  38 x 512 |  38 x  38 x 256 |
|   16 | conv |   512 | 3 x 3 / 1 |   38 x  38 x 256 |  38 x  38 x 512 |
|   17 | max  |       | 2 x 2 / 2 |   38 x  38 x 512 |  19 x  19 x 512 |
|   18 | conv |  1024 | 3 x 3 / 1 |   19 x  19 x 512 |  19 x  19 x1024 |
|   19 | conv |   512 | 1 x 1 / 1 |   19 x  19 x1024 |  19 x  19 x 512 |
|   20 | conv |  1024 | 3 x 3 / 1 |   19 x  19 x 512 |  19 x  19 x1024 |
|   21 | conv |   512 | 1 x 1 / 1 |   19 x  19 x1024 |  19 x  19 x 512 |
|   22 | conv |  1024 | 3 x 3 / 1 |   19 x  19 x 512 |  19 x  19 x1024 |
|   23 | conv |  1024 | 3 x 3 / 1 |   19 x  19 x1024 |  19 x  19 x1024 |
|   24 | conv |  1024 | 3 x 3 / 1 |   19 x  19 x1024 |  19 x  19 x1024 |
|   25 | route|    16 |           |                  |                 |
|   26 | conv |    64 | 1 x 1 / 1 |   38 x  38 x 512 |  38 x  38 x  64 |
|   27 | reorg|       |       / 2 |   38 x  38 x  64 |  19 x  19 x 256 |
|   28 | route| 27 24 |           |                  |                 |
|   29 | conv |  1024 | 3 x 3 / 1 |   19 x  19 x1280 |  19 x  19 x1024 |
|   30 | conv |   425 | 1 x 1 / 1 |   19 x  19 x1024 |  19 x  19 x 425 |
|   31 | detection |  |           |                  |                 |

### YOLO VOC

|      | layer|filters|    size   |       input      |     output      |
|------|------|-------|-----------|------------------|-----------------|
|    0 | conv |    32 | 3 x 3 / 1 |  416 x 416 x   3 | 416 x 416 x  32 |
|    1 | max  |       | 2 x 2 / 2 |  416 x 416 x  32 | 208 x 208 x  32 |
|    2 | conv |    64 | 3 x 3 / 1 |  208 x 208 x  32 | 208 x 208 x  64 |
|    3 | max  |       | 2 x 2 / 2 |  208 x 208 x  64 | 104 x 104 x  64 |
|    4 | conv |   128 | 3 x 3 / 1 |  104 x 104 x  64 | 104 x 104 x 128 |
|    5 | conv |    64 | 1 x 1 / 1 |  104 x 104 x 128 | 104 x 104 x  64 |
|    6 | conv |   128 | 3 x 3 / 1 |  104 x 104 x  64 | 104 x 104 x 128 |
|    7 | max  |       | 2 x 2 / 2 |  104 x 104 x 128 |  52 x  52 x 128 |
|    8 | conv |   256 | 3 x 3 / 1 |   52 x  52 x 128 |  52 x  52 x 256 |
|    9 | conv |   128 | 1 x 1 / 1 |   52 x  52 x 256 |  52 x  52 x 128 |
|   10 | conv |   256 | 3 x 3 / 1 |   52 x  52 x 128 |  52 x  52 x 256 |
|   11 | max  |       | 2 x 2 / 2 |   52 x  52 x 256 |  26 x  26 x 256 |
|   12 | conv |   512 | 3 x 3 / 1 |   26 x  26 x 256 |  26 x  26 x 512 |
|   13 | conv |   256 | 1 x 1 / 1 |   26 x  26 x 512 |  26 x  26 x 256 |
|   14 | conv |   512 | 3 x 3 / 1 |   26 x  26 x 256 |  26 x  26 x 512 |
|   15 | conv |   256 | 1 x 1 / 1 |   26 x  26 x 512 |  26 x  26 x 256 |
|   16 | conv |   512 | 3 x 3 / 1 |   26 x  26 x 256 |  26 x  26 x 512 |
|   17 | max  |       | 2 x 2 / 2 |   26 x  26 x 512 |  13 x  13 x 512 |
|   18 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x 512 |  13 x  13 x1024 |
|   19 | conv |   512 | 1 x 1 / 1 |   13 x  13 x1024 |  13 x  13 x 512 |
|   20 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x 512 |  13 x  13 x1024 |
|   21 | conv |   512 | 1 x 1 / 1 |   13 x  13 x1024 |  13 x  13 x 512 |
|   22 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x 512 |  13 x  13 x1024 |
|   23 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x1024 |  13 x  13 x1024 |
|   24 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x1024 |  13 x  13 x1024 |
|   25 | route|    16 |           |                  |                 |
|   26 | conv |    64 | 1 x 1 / 1 |   26 x  26 x 512 |  26 x  26 x  64 |
|   27 | reorg|       |       / 2 |   26 x  26 x  64 |  13 x  13 x 256 |
|   28 | route| 27 24 |           |                  |                 |
|   29 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x1280 |  13 x  13 x1024 |
|   30 | conv |   125 | 1 x 1 / 1 |   13 x  13 x1024 |  13 x  13 x 125 |
|   31 | detection |  |           |                  |                 |

### Tiny YOLO VOC

|      | layer|filters|    size   |       input      |     output      |
|------|------|-------|-----------|------------------|-----------------|
|    0 | conv |    16 | 3 x 3 / 1 |  416 x 416 x   3 | 416 x 416 x  16 |
|    1 | max  |       | 2 x 2 / 2 |  416 x 416 x  16 | 208 x 208 x  16 |
|    2 | conv |    32 | 3 x 3 / 1 |  208 x 208 x  16 | 208 x 208 x  32 |
|    3 | max  |       | 2 x 2 / 2 |  208 x 208 x  32 | 104 x 104 x  32 |
|    4 | conv |    64 | 3 x 3 / 1 |  104 x 104 x  32 | 104 x 104 x  64 |
|    5 | max  |       | 2 x 2 / 2 |  104 x 104 x  64 |  52 x  52 x  64 |
|    6 | conv |   128 | 3 x 3 / 1 |   52 x  52 x  64 |  52 x  52 x 128 |
|    7 | max  |       | 2 x 2 / 2 |   52 x  52 x 128 |  26 x  26 x 128 |
|    8 | conv |   256 | 3 x 3 / 1 |   26 x  26 x 128 |  26 x  26 x 256 |
|    9 | max  |       | 2 x 2 / 2 |   26 x  26 x 256 |  13 x  13 x 256 |
|   10 | conv |   512 | 3 x 3 / 1 |   13 x  13 x 256 |  13 x  13 x 512 |
|   11 | max  |       | 2 x 2 / 1 |   13 x  13 x 512 |  13 x  13 x 512 |
|   12 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x 512 |  13 x  13 x1024 |
|   13 | conv |  1024 | 3 x 3 / 1 |   13 x  13 x1024 |  13 x  13 x1024 |
|   14 | conv |   125 | 1 x 1 / 1 |   13 x  13 x1024 |  13 x  13 x 125 |
|   15 | detection |  |           |                  |                 |

### YOLO 9000

|      | layer|filters|    size   |       input      |     output      |
|------|------|-------|-----------|------------------|-----------------|
|    0 | conv |    32 | 3 x 3 / 1 |  544 x 544 x   3 | 544 x 544 x  32 |
|    1 | max  |       | 2 x 2 / 2 |  544 x 544 x  32 | 272 x 272 x  32 |
|    2 | conv |    64 | 3 x 3 / 1 |  272 x 272 x  32 | 272 x 272 x  64 |
|    3 | max  |       | 2 x 2 / 2 |  272 x 272 x  64 | 136 x 136 x  64 |
|    4 | conv |   128 | 3 x 3 / 1 |  136 x 136 x  64 | 136 x 136 x 128 |
|    5 | conv |    64 | 1 x 1 / 1 |  136 x 136 x 128 | 136 x 136 x  64 |
|    6 | conv |   128 | 3 x 3 / 1 |  136 x 136 x  64 | 136 x 136 x 128 |
|    7 | max  |       | 2 x 2 / 2 |  136 x 136 x 128 |  68 x  68 x 128 |
|    8 | conv |   256 | 3 x 3 / 1 |   68 x  68 x 128 |  68 x  68 x 256 |
|    9 | conv |   128 | 1 x 1 / 1 |   68 x  68 x 256 |  68 x  68 x 128 |
|   10 | conv |   256 | 3 x 3 / 1 |   68 x  68 x 128 |  68 x  68 x 256 |
|   11 | max  |       | 2 x 2 / 2 |   68 x  68 x 256 |  34 x  34 x 256 |
|   12 | conv |   512 | 3 x 3 / 1 |   34 x  34 x 256 |  34 x  34 x 512 |
|   13 | conv |   256 | 1 x 1 / 1 |   34 x  34 x 512 |  34 x  34 x 256 |
|   14 | conv |   512 | 3 x 3 / 1 |   34 x  34 x 256 |  34 x  34 x 512 |
|   15 | conv |   256 | 1 x 1 / 1 |   34 x  34 x 512 |  34 x  34 x 256 |
|   16 | conv |   512 | 3 x 3 / 1 |   34 x  34 x 256 |  34 x  34 x 512 |
|   17 | max  |       | 2 x 2 / 2 |   34 x  34 x 512 |  17 x  17 x 512 |
|   18 | conv |  1024 | 3 x 3 / 1 |   17 x  17 x 512 |  17 x  17 x1024 |
|   19 | conv |   512 | 1 x 1 / 1 |   17 x  17 x1024 |  17 x  17 x 512 |
|   20 | conv |  1024 | 3 x 3 / 1 |   17 x  17 x 512 |  17 x  17 x1024 |
|   21 | conv |   512 | 1 x 1 / 1 |   17 x  17 x1024 |  17 x  17 x 512 |
|   22 | conv |  1024 | 3 x 3 / 1 |   17 x  17 x 512 |  17 x  17 x1024 |
|   23 | conv | 28269 | 1 x 1 / 1 |   17 x  17 x1024 |  17 x  17 x28269|
|   24 | detection |  |           |                  |                 |
