#include "dropout_layer.h"
#include "params.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer *make_dropout_layer(int batch, int inputs, float probability)
{
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
    dropout_layer *layer = calloc(1, sizeof(dropout_layer));
    layer->probability = probability;
    layer->inputs = inputs;
    layer->batch = batch;
    layer->rand = calloc(inputs*batch, sizeof(float));
    layer->scale = 1./(1.-probability);
    #ifdef GPU
    layer->rand_gpu = cuda_make_array(layer->rand, inputs*batch);
    #endif
    return layer;
} 

void resize_dropout_layer(dropout_layer *layer, int inputs)
{
    layer->rand = realloc(layer->rand, layer->inputs*layer->batch*sizeof(float));
    #ifdef GPU
    cuda_free(layer->rand_gpu);

    layer->rand_gpu = cuda_make_array(layer->rand, inputs*layer->batch);
    #endif
}

void forward_dropout_layer(dropout_layer layer, network_state state)
{
    int i;
    if (!state.train) return;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        float r = rand_uniform();
        layer.rand[i] = r;
        if(r < layer.probability) state.input[i] = 0;
        else state.input[i] *= layer.scale;
    }
}

void backward_dropout_layer(dropout_layer layer, network_state state)
{
    int i;
    if(!state.delta) return;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        float r = layer.rand[i];
        if(r < layer.probability) state.delta[i] = 0;
        else state.delta[i] *= layer.scale;
    }
}

