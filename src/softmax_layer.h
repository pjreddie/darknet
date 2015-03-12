#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "params.h"

typedef struct {
    int inputs;
    int batch;
    int groups;
    float *delta;
    float *output;
    #ifdef GPU
    float * delta_gpu;
    float * output_gpu;
    #endif
} softmax_layer;

void softmax_array(float *input, int n, float *output);
softmax_layer *make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer layer, network_state state);
void backward_softmax_layer(const softmax_layer layer, network_state state);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer layer);
void forward_softmax_layer_gpu(const softmax_layer layer, network_state state);
void backward_softmax_layer_gpu(const softmax_layer layer, network_state state);
#endif

#endif
