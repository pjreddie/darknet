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
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
    int b;
    for(b = 0; b < batch; ++b){
        int i;
        int count = 0;
        for(i = 0; i < hierarchy->groups; ++i){
            int group_size = hierarchy->group_size[i];
            softmax(input+b*inputs + count, group_size, temp, output+b*inputs + count, 1);
            count += group_size;
        }
    }
}

void forward_softmax_layer(const softmax_layer l, network_state state)
{
    int b;
    int inputs = l.inputs / l.groups;
    int batch = l.batch * l.groups;
    if(l.softmax_tree){
        softmax_tree(state.input, batch, inputs, l.temperature, l.softmax_tree, l.output);
    } else {
        for(b = 0; b < batch; ++b){
            softmax(state.input+b*inputs, inputs, l.temperature, l.output+b*inputs, 1);
        }
    }
}

void backward_softmax_layer(const softmax_layer l, network_state state)
{
    int i;
    for(i = 0; i < l.inputs*l.batch; ++i){
        state.delta[i] += l.delta[i];
    }
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network_state state)
{
    int inputs = l.inputs / l.groups;
    int batch = l.batch * l.groups;
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(state.input+count, group_size, inputs, batch, l.temperature, l.output_gpu + count);
            count += group_size;
        }
    } else {
        softmax_gpu(state.input, inputs, inputs, batch, l.temperature, l.output_gpu);
    }
}

void backward_softmax_layer_gpu(const softmax_layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
}

#endif
