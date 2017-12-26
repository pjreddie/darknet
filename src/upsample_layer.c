#include "upsample_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    layer l = {0};
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.stride = stride;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_upsample_layer;
    l.backward = backward_upsample_layer;
    #ifdef GPU
    l.forward_gpu = forward_upsample_layer_gpu;
    l.backward_gpu = backward_upsample_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_upsample_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w*l->stride;
    l->out_h = h*l->stride;
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->h*l->w*l->c;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

void forward_upsample_layer(const layer l, network net)
{
    int i, j, k, b;
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            for(j = 0; j < l.h*l.stride; ++j){
                for(i = 0; i < l.w*l.stride; ++i){
                    int in_index = b*l.inputs + k*l.w*l.h + (j/l.stride)*l.w + i/l.stride;
                    int out_index = b*l.inputs + k*l.w*l.h + j*l.w + i;
                    l.output[out_index] = net.input[in_index];
                }
            }
        }
    }
}

void backward_upsample_layer(const layer l, network net)
{
    int i, j, k, b;
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            for(j = 0; j < l.h*l.stride; ++j){
                for(i = 0; i < l.w*l.stride; ++i){
                    int in_index = b*l.inputs + k*l.w*l.h + (j/l.stride)*l.w + i/l.stride;
                    int out_index = b*l.inputs + k*l.w*l.h + j*l.w + i;
                    net.delta[in_index] += l.delta[out_index];
                }
            }
        }
    }
}

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net)
{
    upsample_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
}

void backward_upsample_layer_gpu(const layer l, network net)
{
    upsample_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.delta_gpu);
}
#endif
