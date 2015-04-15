#include "cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "sse")==0) return SSE;
    fprintf(stderr, "Couldn't find activation function %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SSE:
            return "sse";
    }
    return "sse";
}

cost_layer *make_cost_layer(int batch, int inputs, COST_TYPE type)
{
    fprintf(stderr, "Cost Layer: %d inputs\n", inputs);
    cost_layer *layer = calloc(1, sizeof(cost_layer));
    layer->batch = batch;
    layer->inputs = inputs;
    layer->type = type;
    layer->delta = calloc(inputs*batch, sizeof(float));
    layer->output = calloc(1, sizeof(float));
    #ifdef GPU
    layer->delta_gpu = cuda_make_array(layer->delta, inputs*batch);
    #endif
    return layer;
}

void forward_cost_layer(cost_layer layer, network_state state)
{
    if (!state.truth) return;
    copy_cpu(layer.batch*layer.inputs, state.truth, 1, layer.delta, 1);
    axpy_cpu(layer.batch*layer.inputs, -1, state.input, 1, layer.delta, 1);
    *(layer.output) = dot_cpu(layer.batch*layer.inputs, layer.delta, 1, layer.delta, 1);
    //printf("cost: %f\n", *layer.output);
}

void backward_cost_layer(const cost_layer layer, network_state state)
{
    copy_cpu(layer.batch*layer.inputs, layer.delta, 1, state.delta, 1);
}

#ifdef GPU

void pull_cost_layer(cost_layer layer)
{
    cuda_pull_array(layer.delta_gpu, layer.delta, layer.batch*layer.inputs);
}

void push_cost_layer(cost_layer layer)
{
    cuda_push_array(layer.delta_gpu, layer.delta, layer.batch*layer.inputs);
}

void forward_cost_layer_gpu(cost_layer layer, network_state state)
{
    if (!state.truth) return;
    
    copy_ongpu(layer.batch*layer.inputs, state.truth, 1, layer.delta_gpu, 1);
    axpy_ongpu(layer.batch*layer.inputs, -1, state.input, 1, layer.delta_gpu, 1);

    cuda_pull_array(layer.delta_gpu, layer.delta, layer.batch*layer.inputs);
    *(layer.output) = dot_cpu(layer.batch*layer.inputs, layer.delta, 1, layer.delta, 1);
}

void backward_cost_layer_gpu(const cost_layer layer, network_state state)
{
    copy_ongpu(layer.batch*layer.inputs, layer.delta_gpu, 1, state.delta, 1);
}
#endif

