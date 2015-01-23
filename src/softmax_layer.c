#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include <float.h>
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
    layer->jacobian = calloc(inputs*inputs*batch, sizeof(float));
    #ifdef GPU
    layer->output_gpu = cuda_make_array(layer->output, inputs*batch); 
    layer->delta_gpu = cuda_make_array(layer->delta, inputs*batch); 
    #endif
    return layer;
}

void forward_softmax_layer(const softmax_layer layer, float *input)
{
    int i,b;
    for(b = 0; b < layer.batch; ++b){
        float sum = 0;
        float largest = -FLT_MAX;
        for(i = 0; i < layer.inputs; ++i){
            if(input[i+b*layer.inputs] > largest) largest = input[i+b*layer.inputs];
        }
        for(i = 0; i < layer.inputs; ++i){
            sum += exp(input[i+b*layer.inputs]-largest);
        }
        if(sum) sum = largest+log(sum);
        else sum = largest-100;
        for(i = 0; i < layer.inputs; ++i){
            layer.output[i+b*layer.inputs] = exp(input[i+b*layer.inputs]-sum);
        }
    }
}

void backward_softmax_layer(const softmax_layer layer, float *delta)
{
    int i;
    for(i = 0; i < layer.inputs*layer.batch; ++i){
        delta[i] = layer.delta[i];
    }
}

