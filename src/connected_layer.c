#include "connected_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activator)
{
    printf("Connected Layer: %d inputs, %d outputs\n", inputs, outputs);
    int i;
    connected_layer *layer = calloc(1, sizeof(connected_layer));
    layer->inputs = inputs;
    layer->outputs = outputs;

    layer->output = calloc(outputs, sizeof(double*));
    layer->delta = calloc(outputs, sizeof(double*));

    layer->weight_updates = calloc(inputs*outputs, sizeof(double));
    layer->weight_momentum = calloc(inputs*outputs, sizeof(double));
    layer->weights = calloc(inputs*outputs, sizeof(double));
    for(i = 0; i < inputs*outputs; ++i)
        layer->weights[i] = .01*(.5 - (double)rand()/RAND_MAX);

    layer->bias_updates = calloc(outputs, sizeof(double));
    layer->bias_momentum = calloc(outputs, sizeof(double));
    layer->biases = calloc(outputs, sizeof(double));
    for(i = 0; i < outputs; ++i)
        layer->biases[i] = 1;

    if(activator == SIGMOID){
        layer->activation = sigmoid_activation;
        layer->gradient = sigmoid_gradient;
    }else if(activator == RELU){
        layer->activation = relu_activation;
        layer->gradient = relu_gradient;
    }else if(activator == IDENTITY){
        layer->activation = identity_activation;
        layer->gradient = identity_gradient;
    }

    return layer;
}

void forward_connected_layer(connected_layer layer, double *input)
{
    int i, j;
    for(i = 0; i < layer.outputs; ++i){
        layer.output[i] = layer.biases[i];
        for(j = 0; j < layer.inputs; ++j){
            layer.output[i] += input[j]*layer.weights[i*layer.inputs + j];
        }
        layer.output[i] = layer.activation(layer.output[i]);
    }
}

void learn_connected_layer(connected_layer layer, double *input)
{
    int i, j;
    for(i = 0; i < layer.outputs; ++i){
        layer.bias_updates[i] += layer.delta[i];
        for(j = 0; j < layer.inputs; ++j){
            layer.weight_updates[i*layer.inputs + j] += layer.delta[i]*input[j];
        }
    }
}

void update_connected_layer(connected_layer layer, double step, double momentum, double decay)
{
    int i,j;
    for(i = 0; i < layer.outputs; ++i){
        layer.bias_momentum[i] = step*(layer.bias_updates[i] - decay*layer.biases[i]) + momentum*layer.bias_momentum[i];
        layer.biases[i] += layer.bias_momentum[i];
        for(j = 0; j < layer.inputs; ++j){
            int index = i*layer.inputs+j;
            layer.weight_momentum[index] = step*(layer.weight_updates[index] - decay*layer.weights[index]) + momentum*layer.weight_momentum[index];
            layer.weights[index] += layer.weight_momentum[index];
        }
    }
    memset(layer.bias_updates, 0, layer.outputs*sizeof(double));
    memset(layer.weight_updates, 0, layer.outputs*layer.inputs*sizeof(double));
}

void backward_connected_layer(connected_layer layer, double *input, double *delta)
{
    int i, j;

    for(j = 0; j < layer.inputs; ++j){
        double grad = layer.gradient(input[j]);
        delta[j] = 0;
        for(i = 0; i < layer.outputs; ++i){
            delta[j] += layer.delta[i]*layer.weights[i*layer.inputs + j];
        }
        delta[j] *= grad;
    }
}

