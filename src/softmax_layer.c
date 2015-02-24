#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer *make_softmax_layer(int batch, int groups, int inputs)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "Softmax Layer: %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->batch = batch;
    layer->groups = groups;
    layer->inputs = inputs;
    layer->output = calloc(inputs*batch, sizeof(float));
    layer->delta = calloc(inputs*batch, sizeof(float));
    #ifdef GPU
    layer->output_gpu = cuda_make_array(layer->output, inputs*batch); 
    layer->delta_gpu = cuda_make_array(layer->delta, inputs*batch); 
    #endif
    return layer;
}

void softmax_array(float *input, int n, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        sum += exp(input[i]-largest);
    }
    if(sum) sum = largest+log(sum);
    else sum = largest-100;
    for(i = 0; i < n; ++i){
        output[i] = exp(input[i]-sum);
    }
}

void forward_softmax_layer(const softmax_layer layer, float *input)
{
    int b;
    int inputs = layer.inputs / layer.groups;
    int batch = layer.batch * layer.groups;
    for(b = 0; b < batch; ++b){
        softmax_array(input+b*inputs, inputs, layer.output+b*inputs);
    }
}

void backward_softmax_layer(const softmax_layer layer, float *delta)
{
    int i;
    for(i = 0; i < layer.inputs*layer.batch; ++i){
        delta[i] = layer.delta[i];
    }
}

