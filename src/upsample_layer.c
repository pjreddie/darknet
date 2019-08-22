#include "upsample_layer.h"
#include "dark_cuda.h"
#include "blas.h"

#include <stdio.h>

layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    if(stride < 0){
        stride = -stride;
        l.reverse=1;
        l.out_w = w/stride;
        l.out_h = h/stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta = (float*)calloc(l.outputs * batch, sizeof(float));
    l.output = (float*)calloc(l.outputs * batch, sizeof(float));

    l.forward = forward_upsample_layer;
    l.backward = backward_upsample_layer;
    #ifdef GPU
    l.forward_gpu = forward_upsample_layer_gpu;
    l.backward_gpu = backward_upsample_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    if(l.reverse) fprintf(stderr, "downsample              %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else fprintf(stderr, "upsample                %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_upsample_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w*l->stride;
    l->out_h = h*l->stride;
    if(l->reverse){
        l->out_w = w/l->stride;
        l->out_h = h/l->stride;
    }
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->h*l->w*l->c;
    l->delta = (float*)realloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif

}

void forward_upsample_layer(const layer l, network_state net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.reverse){
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }else{
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}

void backward_upsample_layer(const layer l, network_state state)
{
    if(l.reverse){
        upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, state.delta);
    }else{
        upsample_cpu(state.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
    }
}

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network_state state)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.reverse){
        upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, state.input);
    }else{
        upsample_gpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
    }
}

void backward_upsample_layer_gpu(const layer l, network_state state)
{
    if(l.reverse){
        upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, state.delta);
    }else{
        upsample_gpu(state.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu);
    }
}
#endif
