#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "opencl.h"

typedef struct {
    int inputs;
    int batch;
    float *delta;
    float *output;
    #ifdef GPU
    cl_mem delta_cl;
    #endif
} cost_layer;

cost_layer *make_cost_layer(int batch, int inputs);
void forward_cost_layer(const cost_layer layer, float *input, float *truth);
void backward_cost_layer(const cost_layer layer, float *input, float *delta);

#ifdef GPU
void forward_cost_layer_gpu(cost_layer layer, cl_mem input, cl_mem truth);
void backward_cost_layer_gpu(const cost_layer layer, cl_mem input, cl_mem delta);
#endif

#endif
