![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Added features
Darknet is now able to export image annotations into JSON files (see `annotations/annotation_example.json`). Moreover, it can now deal with an images list given in parameter to annotate images in a row. For instance :
```
$ ./darknet detector test cfg/coco.data cfg/yolo.cfg weights/yolo.weights data/dog.jpg data/horses.jpg
```

# Darknet
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
