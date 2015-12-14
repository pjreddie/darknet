#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer detection_layer;

detection_layer make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
void forward_detection_layer(const detection_layer l, network_state state);
void backward_detection_layer(const detection_layer l, network_state state);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network_state state);
void backward_detection_layer_gpu(detection_layer l, network_state state);
#endif

#endif
