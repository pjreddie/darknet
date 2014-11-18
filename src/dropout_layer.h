#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H
#include "opencl.h"

typedef struct{
    int batch;
    int inputs;
    float probability;
    #ifdef GPU
    float *rand;
    cl_mem rand_cl;
    #endif
} dropout_layer;

dropout_layer *make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer layer, float *input);
void backward_dropout_layer(dropout_layer layer, float *input, float *delta);
    #ifdef GPU
void forward_dropout_layer_gpu(dropout_layer layer, cl_mem input);

#endif
#endif
