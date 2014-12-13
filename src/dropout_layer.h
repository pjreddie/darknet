#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H
#include "opencl.h"

typedef struct{
    int batch;
    int inputs;
    float probability;
    float scale;
    float *rand;
    #ifdef GPU
    cl_mem rand_cl;
    #endif
} dropout_layer;

dropout_layer *make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer layer, float *input);
void backward_dropout_layer(dropout_layer layer, float *delta);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer layer, cl_mem input);
void backward_dropout_layer_gpu(dropout_layer layer, cl_mem delta);

#endif
#endif
