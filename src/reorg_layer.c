#include "reorg_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>


layer make_reorg_layer(int batch, int h, int w, int c, int stride, int reverse)
{
    layer l = {0};
    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.h = h;
    l.w = w;
    l.c = c;
    if(reverse){
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c/(stride*stride);
    }else{
        l.out_w = w/stride;
        l.out_h = h/stride;
        l.out_c = c*(stride*stride);
    }
    fprintf(stderr, "Reorg Layer: %d x %d x %d image -> %d x %d x %d image, \n", w,h,c,l.out_w, l.out_h, l.out_c);
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));

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

    l->h = h;
    l->w = w;

    l->out_w = w*stride;
    l->out_h = h*stride;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}

void forward_reorg_layer(const layer l, network_state state)
{
    int b,i,j,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            for(j = 0; j < l.h; ++j){
                for(i = 0; i < l.w; ++i){
                    int in_index  = i + l.w*(j + l.h*(k + l.c*b));

                    int c2 = k % l.out_c;
                    int offset = k / l.out_c;
                    int w2 = i*l.stride + offset % l.stride;
                    int h2 = j*l.stride + offset / l.stride;
                    int out_index = w2 + l.out_w*(h2 + l.out_h*(c2 + l.out_c*b));
                    l.output[out_index] = state.input[in_index];
                }
            }
        }
    }
}

void backward_reorg_layer(const layer l, network_state state)
{
    int b,i,j,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            for(j = 0; j < l.h; ++j){
                for(i = 0; i < l.w; ++i){
                    int in_index  = i + l.w*(j + l.h*(k + l.c*b));

                    int c2 = k % l.out_c;
                    int offset = k / l.out_c;
                    int w2 = i*l.stride + offset % l.stride;
                    int h2 = j*l.stride + offset / l.stride;
                    int out_index = w2 + l.out_w*(h2 + l.out_h*(c2 + l.out_c*b));
                    state.delta[in_index] = l.delta[out_index];
                }
            }
        }
    }
}

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network_state state)
{
    if(l.reverse){
        reorg_ongpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
    }else {
        reorg_ongpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
    }
}

void backward_reorg_layer_gpu(layer l, network_state state)
{
    if(l.reverse){
        reorg_ongpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, state.delta);
    }else{
        reorg_ongpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, state.delta);
    }
}
#endif
