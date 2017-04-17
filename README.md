![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## About this version
This version of darknet has been modified to return a custom data representation. The only network that has been modified is the YOLO one. 
If you want to use our custom version of the code you should consider the following changes:

    Original: ./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg
    Custom:   ./darknet detector test2 cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg
    
You may noticed that instead of running the 'test', you will be running the 'test2'.
    
The purpose of the modification is to get all the information about the bounding boxes that YOLO detetects.
This version of darkent is used in the [SR-Clustering](https://github.com/MarcBS/SR-Clustering/tree/SR-Clustering-w/-YOLO) project at the University of Barcelona.

You may need yolo9000.weights. You can download it from [here](http://pjreddie.com/media/files/yolo9000.weights).
