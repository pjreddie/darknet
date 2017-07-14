//
// Created by frivas on 25/04/16.
//

#ifndef DARKNET_YOLO_H
#define DARKNET_YOLO_H

#include "box.h"



void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);


#endif //DARKNET_YOLO_H
