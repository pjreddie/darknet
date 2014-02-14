#include "softmax_layer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

softmax_layer *make_softmax_layer(int inputs)
{
    fprintf(stderr, "Softmax Layer: %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->inputs = inputs;
    layer->output = calloc(inputs, sizeof(float));
    layer->delta = calloc(inputs, sizeof(float));
    return layer;
}

/* UNSTABLE!
void forward_softmax_layer(const softmax_layer layer, float *input)
{
    int i;
    float sum = 0;
    for(i = 0; i < layer.inputs; ++i){
        sum += exp(input[i]);
    }
    for(i = 0; i < layer.inputs; ++i){
        layer.output[i] = exp(input[i])/sum;
    }
}
*/
void forward_softmax_layer(const softmax_layer layer, float *input)
{
    int i;
    float sum = 0;
    float largest = 0;
    for(i = 0; i < layer.inputs; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < layer.inputs; ++i){
        sum += exp(input[i]-largest);
        printf("%f, ", input[i]);
    }
    printf("\n");
    if(sum) sum = largest+log(sum);
    else sum = largest-100;
    for(i = 0; i < layer.inputs; ++i){
        layer.output[i] = exp(input[i]-sum);
    }
}

void backward_softmax_layer(const softmax_layer layer, float *input, float *delta)
{
    int i;
    for(i = 0; i < layer.inputs; ++i){
        delta[i] = layer.delta[i];
    }
}

