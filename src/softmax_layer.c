#include "softmax_layer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

softmax_layer *make_softmax_layer(int inputs)
{
    printf("Softmax Layer: %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->inputs = inputs;
    layer->output = calloc(inputs, sizeof(double));
    layer->delta = calloc(inputs, sizeof(double));
    return layer;
}

void forward_softmax_layer(const softmax_layer layer, double *input)
{
    int i;
    double sum = 0;
    for(i = 0; i < layer.inputs; ++i){
        sum += exp(input[i]);
    }
    for(i = 0; i < layer.inputs; ++i){
        layer.output[i] = exp(input[i])/sum;
    }
}

void backward_softmax_layer(const softmax_layer layer, double *input, double *delta)
{
    int i;
    for(i = 0; i < layer.inputs; ++i){
        delta[i] = layer.delta[i];
    }
}

