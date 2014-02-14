#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"

typedef struct{
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

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activation);

void forward_connected_layer(connected_layer layer, float *input);
void backward_connected_layer(connected_layer layer, float *input, float *delta);
void learn_connected_layer(connected_layer layer, float *input);
void update_connected_layer(connected_layer layer, float step, float momentum, float decay);


#endif

