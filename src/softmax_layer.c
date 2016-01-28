#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "Softmax Layer: %d inputs\n", inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    #ifdef GPU
    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void softmax_array(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        sum += exp(input[i]/temp-largest/temp);
    }
    if(sum) sum = largest/temp+log(sum);
    else sum = largest-100;
    for(i = 0; i < n; ++i){
        output[i] = exp(input[i]/temp-sum);
    }
}

void forward_softmax_layer(const softmax_layer l, network_state state)
{
    int b;
    int inputs = l.inputs / l.groups;
    int batch = l.batch * l.groups;
    for(b = 0; b < batch; ++b){
        softmax_array(state.input+b*inputs, inputs, l.temperature, l.output+b*inputs);
    }
}

void backward_softmax_layer(const softmax_layer l, network_state state)
{
    int i;
    for(i = 0; i < l.inputs*l.batch; ++i){
        state.delta[i] += l.delta[i];
    }
}

