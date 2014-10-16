#include "maxpool_layer.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    return float_to_image(h,w,c,layer.output);
}

image get_maxpool_delta(maxpool_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    return float_to_image(h,w,c,layer.delta);
}

maxpool_layer *make_maxpool_layer(int batch, int h, int w, int c, int size, int stride)
{
    fprintf(stderr, "Maxpool Layer: %d x %d x %d image, %d size, %d stride\n", h,w,c,size,stride);
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->batch = batch;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->size = size;
    layer->stride = stride;
    layer->indexes = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c*batch, sizeof(int));
    layer->output = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c*batch, sizeof(float));
    layer->delta = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c*batch, sizeof(float));
    return layer;
}

void resize_maxpool_layer(maxpool_layer *layer, int h, int w, int c)
{
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->output = realloc(layer->output, ((h-1)/layer->stride+1) * ((w-1)/layer->stride+1) * c * layer->batch* sizeof(float));
    layer->delta = realloc(layer->delta, ((h-1)/layer->stride+1) * ((w-1)/layer->stride+1) * c * layer->batch*sizeof(float));
}

void forward_maxpool_layer(const maxpool_layer layer, float *input)
{
    int b,i,j,k,l,m;
    int w_offset = (-layer.size-1)/2 + 1;
    int h_offset = (-layer.size-1)/2 + 1;

    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;

    for(b = 0; b < layer.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(l = 0; l < layer.size; ++l){
                        for(m = 0; m < layer.size; ++m){
                            int cur_h = h_offset + i*layer.stride + l;
                            int cur_w = w_offset + j*layer.stride + m;
                            int index = cur_w + layer.w*(cur_h + layer.h*(k + b*layer.c));
                            int valid = (cur_h >= 0 && cur_h < layer.h &&
                                         cur_w >= 0 && cur_w < layer.w);
                            float val = (valid != 0) ? input[index] : -INFINITY;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    layer.output[out_index] = max;
                    layer.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer layer, float *input, float *delta)
{
    int i;
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    memset(delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));
    for(i = 0; i < h*w*c*layer.batch; ++i){
        int index = layer.indexes[i];
        delta[index] += layer.delta[i];
    }
}

