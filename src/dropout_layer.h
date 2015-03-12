#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H
#include "params.h"

typedef struct{
    int batch;
    int inputs;
    float probability;
    float scale;
    float *rand;
    #ifdef GPU
    float * rand_gpu;
    #endif
} dropout_layer;

dropout_layer *make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer layer, network_state state);
void backward_dropout_layer(dropout_layer layer, network_state state);
void resize_dropout_layer(dropout_layer *layer, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer layer, network_state state);
void backward_dropout_layer_gpu(dropout_layer layer, network_state state);

#endif
#endif
