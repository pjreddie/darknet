#include "upsample_dla_layer.h"
#include "blas.h"

#include <stdio.h>

layer make_upsample_dla_layer(int batch, int w, int h, int c, int stride)
{
    layer l = {0};
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
    l.outputs = l.out_w*l.out_h*l.out_c/ATOMIC_CUBE;
    l.inputs = l.w*l.h*l.c;
    l.delta =  calloc(l.outputs*batch, ATOMIC_CUBE);
    l.output = calloc(l.outputs*batch, ATOMIC_CUBE);;

    l.forward = forward_upsample_dla_layer;

    if(l.reverse) fprintf(stderr, "downsample_dla     %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else fprintf(stderr, "upsample_dla       %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void *cubecpy(void *dst, const void *src){
    char *tmp_dst= dst;
    const char *tmp_src = src;
    int byte = ATOMIC_CUBE;

    while(byte) {
        *tmp_dst++ = *tmp_src++;
        byte--;
    }

    return dst;
}

void upsample_dla(int8_t *in, int w, int h, int c, int batch, int stride, int forward, int8_t *out)
{
    int i, j, k, b;

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c*ATOMIC_CUBE + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride*ATOMIC_CUBE + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) cubecpy(out + out_index, in + in_index);
                }
            }
        }
    }
}

void forward_upsample_dla_layer(const layer l, network net)
{
    if(l.reverse){
        upsample_dla(l.output_i8, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, net.input_i8);
    }else{
        upsample_dla(net.input_i8, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_i8);
    }
}
