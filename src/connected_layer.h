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

    float *weight_adapt;
    float *bias_adapt;

    float *weight_momentum;
    float *bias_momentum;

    float *output;
    float *delta;
    
    ACTIVATION activation;

} connected_layer;

connected_layer *make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, float learning_rate, float momentum, float decay);

void forward_connected_layer(connected_layer layer, float *input);
void backward_connected_layer(connected_layer layer, float *input, float *delta);
void update_connected_layer(connected_layer layer);


#endif

