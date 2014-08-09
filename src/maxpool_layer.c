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
    layer->max_indexes = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c*batch, sizeof(int));
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
    int b;
    for(b = 0; b < layer.batch; ++b){
        int h = (layer.h-1)/layer.stride + 1;
        int w = (layer.w-1)/layer.stride + 1;
        int c = layer.c;

        int i,j,k,l,m;
        for(k = 0; k < layer.c; ++k){
            for(i = 0; i < layer.h; i += layer.stride){
                for(j = 0; j < layer.w; j += layer.stride){
                    int out_index = j/layer.stride + w*(i/layer.stride + h*(k + c*b));
                    layer.output[out_index] = -FLT_MAX;
                    int lower = (-layer.size-1)/2 + 1;
                    int upper = layer.size/2 + 1;

                    int lh = (i+lower < 0)       ? 0 : i+lower;
                    int uh = (i+upper > layer.h) ? layer.h : i+upper;

                    int lw = (j+lower < 0)       ? 0 : j+lower;
                    int uw = (j+upper > layer.w) ? layer.w : j+upper;
                    for(l = lh; l < uh; ++l){
                        for(m = lw; m < uw; ++m){
                            //printf("%d %d\n", l, m);
                            int index = m + layer.w*(l + layer.h*(k + b*layer.c));
                            if(input[index] > layer.output[out_index]){
                                layer.output[out_index] = input[index];
                                layer.max_indexes[out_index] = index;
                            }
                        }
                    }
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
        int index = layer.max_indexes[i];
        delta[index] += layer.delta[i];
    }
}

