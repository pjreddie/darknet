#include "softmax_layer.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

softmax_layer *make_softmax_layer(int batch, int inputs)
{
    fprintf(stderr, "Softmax Layer: %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->batch = batch;
    layer->inputs = inputs;
    layer->output = calloc(inputs*batch, sizeof(float));
    layer->delta = calloc(inputs*batch, sizeof(float));
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
    int i,b;
    for(b = 0; b < layer.batch; ++b){
        float sum = 0;
        float largest = 0;
        for(i = 0; i < layer.inputs; ++i){
            if(input[i+b*layer.inputs] > largest) largest = input[i+b*layer.inputs];
        }
        for(i = 0; i < layer.inputs; ++i){
            sum += exp(input[i+b*layer.inputs]-largest);
            //printf("%f, ", input[i]);
        }
        //printf("\n");
        if(sum) sum = largest+log(sum);
        else sum = largest-100;
        for(i = 0; i < layer.inputs; ++i){
            layer.output[i+b*layer.inputs] = exp(input[i+b*layer.inputs]-sum);
        }
    }
}

void backward_softmax_layer(const softmax_layer layer, float *input, float *delta)
{
    int i;
    for(i = 0; i < layer.inputs*layer.batch; ++i){
        delta[i] = layer.delta[i];
    }
}

