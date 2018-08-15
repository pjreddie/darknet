#include "upsample_odla_layer.h"
#include "blas.h"

#include <stdio.h>

layer make_upsample_odla_layer(int batch, int w, int h, int c, int stride, int output_layer, int tensor)
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
    l.stride = stride;
    l.upsample_output_layer = output_layer;
    l.upsample_output_tensor = tensor;
    l.reverse = 0;

    l.forward = forward_upsample_odla_layer;

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

void forward_upsample_odla_layer(const layer l, network net)
{
    int8_t *output;
    layer *output_layer;

    output_layer = &net.layers[l.upsample_output_layer];
    output = output_layer->output_tensors[l.upsample_output_tensor].buffer;

    upsample_dla(net.input_i8, l.w, l.h, l.c, l.batch, l.stride, 1, output);
}
