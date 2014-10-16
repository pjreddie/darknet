#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "opencl.h"

typedef struct{
    float learning_rate;
    float momentum;
    float decay;

    int batch;
    int inputs;
    int outputs;
    float *weights;
    float *biases;

    float *weight_updates;
    float *bias_updates;

    float *weight_adapt;
    float *bias_adapt;

    float *output;
    float *delta;
    
    #ifdef GPU
    cl_mem weights_cl;
    cl_mem biases_cl;

    cl_mem weight_updates_cl;
    cl_mem bias_updates_cl;

    cl_mem output_cl;
    cl_mem delta_cl;
    #endif
    ACTIVATION activation;

} connected_layer;

connected_layer *make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, float learning_rate, float momentum, float decay);

void forward_connected_layer(connected_layer layer, float *input);
void backward_connected_layer(connected_layer layer, float *input, float *delta);
void update_connected_layer(connected_layer layer);

#ifdef GPU
void forward_connected_layer_gpu(connected_layer layer, cl_mem input);
void backward_connected_layer_gpu(connected_layer layer, cl_mem input, cl_mem delta);
void update_connected_layer_gpu(connected_layer layer);
#endif

#endif

