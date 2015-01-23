#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"

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

    float *weight_prev;
    float *bias_prev;

    float *output;
    float *delta;
    
    #ifdef GPU
    float * weights_gpu;
    float * biases_gpu;

    float * weight_updates_gpu;
    float * bias_updates_gpu;

    float * output_gpu;
    float * delta_gpu;
    #endif
    ACTIVATION activation;

} connected_layer;

void secret_update_connected_layer(connected_layer *layer);
connected_layer *make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, float learning_rate, float momentum, float decay);

void forward_connected_layer(connected_layer layer, float *input);
void backward_connected_layer(connected_layer layer, float *input, float *delta);
void update_connected_layer(connected_layer layer);

#ifdef GPU
void forward_connected_layer_gpu(connected_layer layer, float * input);
void backward_connected_layer_gpu(connected_layer layer, float * input, float * delta);
void update_connected_layer_gpu(connected_layer layer);
void push_connected_layer(connected_layer layer);
void pull_connected_layer(connected_layer layer);
#endif

#endif

