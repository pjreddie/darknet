![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

Yolo v4 paper: https://arxiv.org/abs/2004.10934

Yolo v4 source code: https://github.com/AlexeyAB/darknet

Useful links: https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Docker Support ##
You can use the Darknet directly using Docker.

### Requirements
* Debian-based OS (Tested on [Ubuntu 20.04 LTS](https://ubuntu.com/download))
* [Docker](https://www.docker.com/)
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* NVIDIA GPU with CUDA-Support (Tested on NVIDIA GeForce GTX 1050 Mobile)

### Getting started
* Clone the repository
* Copy the .env.example as .env from docker directory
    * Edit the .env file
        * The most important (required) variables are the following:
            * DOCKER_APP_UID, Must be the host user ID, example `1000`
            * DOCKER_APP_GID, Must be the host group ID, example `1000`
            * DOCKER_CAMERA_DEVICE, Must be your webcam device, example `video0`
* Go to project directory with terminal
* Pick and Build your desired Docker image, using the right shell script `*.sh`
    * `./docker/build-cpu-image.sh` to use Darknet with CPU-only Support with OpenCV and OpenMP
    * `./docker/build-gpu-image.sh` to use Darknet with GPU-CUDA Support with OpenCV and OpenMP
    * `./docker/build-images.sh` to build both images
* Download the `yolov3-tiny.weights` and put the file in weights directory
* Edit the Makefile variables to enable CUDA, CUDNN, OpenCV and/or OpenMP
* Pick and Start Docker container, using the right shell script `*.sh`
    * `./docker/access-cpu-container.sh` to use Darknet with CPU-only Support with OpenCV and OpenMP
    * `./docker/access-gpu-container.sh` to use Darknet with GPU-CUDA Support with OpenCV and OpenMP
* Build the Darknet with the following command `make`
* Run the Eagle Test using OpenCV `./darknet imtest data/eagle.jpg`
* Run the YOLO using Webcam `./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights`

### Credits for Docker Support
Developed by IEEE Robotics and Automation Society Student Branch Chapter of International Hellenic University (Serres) as part of his research into robotics, machine learning and machine vision under [YOLO License](./README.md)