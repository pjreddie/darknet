#include "maxpool_layer.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    return double_to_image(h,w,c,layer.output);
}

image get_maxpool_delta(maxpool_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    return double_to_image(h,w,c,layer.delta);
}

maxpool_layer *make_maxpool_layer(int h, int w, int c, int stride)
{
    printf("Maxpool Layer: %d x %d x %d image, %d stride\n", h,w,c,stride);
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->stride = stride;
    layer->output = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c, sizeof(double));
    layer->delta = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c, sizeof(double));
    return layer;
}

void forward_maxpool_layer(const maxpool_layer layer, double *in)
{
    image input = double_to_image(layer.h, layer.w, layer.c, in);
    image output = get_maxpool_image(layer);
    int i,j,k;
    for(i = 0; i < output.h*output.w*output.c; ++i) output.data[i] = -DBL_MAX;
    for(k = 0; k < input.c; ++k){
        for(i = 0; i < input.h; ++i){
            for(j = 0; j < input.w; ++j){
                double val = get_pixel(input, i, j, k);
                double cur = get_pixel(output, i/layer.stride, j/layer.stride, k);
                if(val > cur) set_pixel(output, i/layer.stride, j/layer.stride, k, val);
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer layer, double *in, double *delta)
{
    image input = double_to_image(layer.h, layer.w, layer.c, in);
    image input_delta = double_to_image(layer.h, layer.w, layer.c, delta);
    image output_delta = get_maxpool_delta(layer);
    image output = get_maxpool_image(layer);
    int i,j,k;
    for(k = 0; k < input.c; ++k){
        for(i = 0; i < input.h; ++i){
            for(j = 0; j < input.w; ++j){
                double val = get_pixel(input, i, j, k);
                double cur = get_pixel(output, i/layer.stride, j/layer.stride, k);
                double d = get_pixel(output_delta, i/layer.stride, j/layer.stride, k);
                if(val == cur) {
                    set_pixel(input_delta, i, j, k, d);
                }
                else set_pixel(input_delta, i, j, k, 0);
            }
        }
    }
}

