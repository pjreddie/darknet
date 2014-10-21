#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "opencl.h"

typedef struct {
    int inputs;
    int batch;
    float *delta;
    float *output;
    float *jacobian;
    #ifdef GPU
    cl_mem delta_cl;
    cl_mem output_cl;
    #endif
} softmax_layer;

softmax_layer *make_softmax_layer(int batch, int inputs);
void forward_softmax_layer(const softmax_layer layer, float *input);
void backward_softmax_layer(const softmax_layer layer, float *delta);

#ifdef GPU
void forward_softmax_layer_gpu(const softmax_layer layer, cl_mem input);
void backward_softmax_layer_gpu(const softmax_layer layer, cl_mem delta);
#endif

#endif
