![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

# Usage
## Detection
```
./darknet detect <config-file> <weights-file> <image-to-process> [-thresh <threshold>]
```

Change paths according to description. 
The darknet program will then process the image, output class probabilities a and using OpenCV generate a predictions.png file with bounding boxes.
Default threshold is 25%

Example:
```
./darknet detect cfg/yolov2.cfg yolov2.weights data/dog.jpg
```

Alternative command:
```
./darknet detector test ...
```
## Training
