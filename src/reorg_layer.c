#include "reorg_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>


layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
    layer l = {};
    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.extra = extra;
    l.h = h;
    l.w = w;
    l.c = c;
    l.flatten = flatten;
    if(reverse){
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c/(stride*stride);
    }else{
        l.out_w = w/stride;
        l.out_h = h/stride;
        l.out_c = c*(stride*stride);
    }
    l.reverse = reverse;

    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;

    if(l.extra){
        l.out_w = l.out_h = l.out_c = 0;
        l.outputs = l.inputs + l.extra;
    }

    if(extra){
        fprintf(stderr, "reorg              %4d   ->  %4d\n",  l.inputs, l.outputs);
    } else {
        fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, l.out_w, l.out_h, l.out_c);
    }
    int output_size = l.outputs * batch;
    l.output =  (float*)calloc(output_size, sizeof(float));
    l.delta =   (float*)calloc(output_size, sizeof(float));


    l.forward = forward_reorg_layer;
    l.backward = backward_reorg_layer;
#ifdef GPU
    l.forward_gpu = forward_reorg_layer_gpu;
    l.backward_gpu = backward_reorg_layer_gpu;

    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
    return l;
}

void resize_reorg_layer(layer *l, int w, int h)
{
    int stride = l->stride;
    int c = l->c;

    l->h = h;
    l->w = w;

    if(l->reverse){
        l->out_w = w*stride;
        l->out_h = h*stride;
        l->out_c = c/(stride*stride);
    }else{
        l->out_w = w/stride;
        l->out_h = h/stride;
        l->out_c = c*(stride*stride);
    }

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = (float*)realloc(l->output, output_size * sizeof(float));
    l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}

void forward_reorg_layer(const layer l, network_state state)
{
    int i;
    if(l.flatten){
        memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
        if(l.reverse){
            flatten(l.output, l.w*l.h, l.c, l.batch, 0);
        }else{
            flatten(l.output, l.w*l.h, l.c, l.batch, 1);
        }
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_cpu(l.inputs, state.input + i*l.inputs, 1, l.output + i*l.outputs, 1);
        }
    } else if (l.reverse){
        reorg_cpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
    } else {
        reorg_cpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
    }
}

void backward_reorg_layer(const layer l, network_state state)
{
    int i;
    if(l.flatten){
        memcpy(state.delta, l.delta, l.outputs*l.batch*sizeof(float));
        if(l.reverse){
            flatten(state.delta, l.w*l.h, l.c, l.batch, 1);
        }else{
            flatten(state.delta, l.w*l.h, l.c, l.batch, 0);
        }
    } else if(l.reverse){
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, state.delta);
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_cpu(l.inputs, l.delta + i*l.outputs, 1, state.delta + i*l.inputs, 1);
        }
    }else{
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, state.delta);
    }
}

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network_state state)
{
    int i;
    if(l.flatten){
        if(l.reverse){
            flatten_ongpu(state.input, l.w*l.h, l.c, l.batch, 0, l.output_gpu);
        }else{
            flatten_ongpu(state.input, l.w*l.h, l.c, l.batch, 1, l.output_gpu);
        }
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_ongpu(l.inputs, state.input + i*l.inputs, 1, l.output_gpu + i*l.outputs, 1);
        }
    } else if (l.reverse) {
        reorg_ongpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
    }else {
        reorg_ongpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
    }
}

void backward_reorg_layer_gpu(layer l, network_state state)
{
    if(l.flatten){
        if(l.reverse){
            flatten_ongpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 1, state.delta);
        }else{
            flatten_ongpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 0, state.delta);
        }
    } else if (l.extra) {
        int i;
        for(i = 0; i < l.batch; ++i){
            copy_ongpu(l.inputs, l.delta_gpu + i*l.outputs, 1, state.delta + i*l.inputs, 1);
        }
    } else if(l.reverse){
        reorg_ongpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, state.delta);
    } else {
        reorg_ongpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, state.delta);
    }
}
#endif
