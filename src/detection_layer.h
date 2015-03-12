#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

#include "params.h"

typedef struct {
    int batch;
    int inputs;
    int classes;
    int coords;
    int rescore;
    float *output;
    float *delta;
    #ifdef GPU
    float * output_gpu;
    float * delta_gpu;
    #endif
} detection_layer;

detection_layer *make_detection_layer(int batch, int inputs, int classes, int coords, int rescore);
void forward_detection_layer(const detection_layer layer, network_state state);
void backward_detection_layer(const detection_layer layer, network_state state);
int get_detection_layer_output_size(detection_layer layer);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer layer, network_state state);
void backward_detection_layer_gpu(detection_layer layer, network_state state);
#endif

#endif
