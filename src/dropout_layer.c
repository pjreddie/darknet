#include "dropout_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability, int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c)
{
    dropout_layer l = { (LAYER_TYPE)0 };
    l.type = DROPOUT;
    l.probability = probability;
    l.dropblock = dropblock;
    l.dropblock_size_rel = dropblock_size_rel;
    l.dropblock_size_abs = dropblock_size_abs;
    if (l.dropblock) {
        l.out_w = l.w = w;
        l.out_h = l.h = h;
        l.out_c = l.c = c;

        if (l.w <= 0 || l.h <= 0 || l.c <= 0) {
            printf(" Error: DropBlock - there must be positive values for: l.w=%d, l.h=%d, l.c=%d \n", l.w, l.h, l.c);
            exit(0);
        }
    }
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = (float*)xcalloc(inputs * batch, sizeof(float));
    l.scale = 1./(1.0 - probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
#ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    if (l.dropblock) {
        l.drop_blocks_scale = cuda_make_array_pinned(l.rand, l.batch);
        l.drop_blocks_scale_gpu = cuda_make_array(l.rand, l.batch);
    }
#endif
    if (l.dropblock) {
        if(l.dropblock_size_abs) fprintf(stderr, "dropblock    p = %.3f   l.dropblock_size_abs = %d    %4d  ->   %4d\n", probability, l.dropblock_size_abs, inputs, inputs);
        else fprintf(stderr, "dropblock    p = %.3f   l.dropblock_size_rel = %.2f    %4d  ->   %4d\n", probability, l.dropblock_size_rel, inputs, inputs);
    }
    else fprintf(stderr, "dropout    p = %.3f        %4d  ->   %4d\n", probability, inputs, inputs);
    return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->inputs = l->outputs = inputs;
    l->rand = (float*)xrealloc(l->rand, l->inputs * l->batch * sizeof(float));
#ifdef GPU
    cuda_free(l->rand_gpu);
    l->rand_gpu = cuda_make_array(l->rand, l->inputs*l->batch);

    if (l->dropblock) {
        cudaFreeHost(l->drop_blocks_scale);
        l->drop_blocks_scale = cuda_make_array_pinned(l->rand, l->batch);

        cuda_free(l->drop_blocks_scale_gpu);
        l->drop_blocks_scale_gpu = cuda_make_array(l->rand, l->batch);
    }
#endif
}

void forward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
    if (!state.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) state.input[i] = 0;
        else state.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
    if(!state.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) state.delta[i] = 0;
        else state.delta[i] *= l.scale;
    }
}
