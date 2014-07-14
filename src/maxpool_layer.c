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

maxpool_layer *make_maxpool_layer(int batch, int h, int w, int c, int stride)
{
    fprintf(stderr, "Maxpool Layer: %d x %d x %d image, %d stride\n", h,w,c,stride);
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->batch = batch;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->stride = stride;
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

void forward_maxpool_layer(const maxpool_layer layer, float *in)
{
    int b;
    for(b = 0; b < layer.batch; ++b){
        image input = float_to_image(layer.h, layer.w, layer.c, in+b*layer.h*layer.w*layer.c);

        int h = (layer.h-1)/layer.stride + 1;
        int w = (layer.w-1)/layer.stride + 1;
        int c = layer.c;
        image output = float_to_image(h,w,c,layer.output+b*h*w*c);

        int i,j,k;
        for(i = 0; i < output.h*output.w*output.c; ++i) output.data[i] = -DBL_MAX;
        for(k = 0; k < input.c; ++k){
            for(i = 0; i < input.h; ++i){
                for(j = 0; j < input.w; ++j){
                    float val = get_pixel(input, i, j, k);
                    float cur = get_pixel(output, i/layer.stride, j/layer.stride, k);
                    if(val > cur) set_pixel(output, i/layer.stride, j/layer.stride, k, val);
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer layer, float *in, float *delta)
{
    int b;
    for(b = 0; b < layer.batch; ++b){
        image input = float_to_image(layer.h, layer.w, layer.c, in+b*layer.h*layer.w*layer.c);
        image input_delta = float_to_image(layer.h, layer.w, layer.c, delta+b*layer.h*layer.w*layer.c);
        int h = (layer.h-1)/layer.stride + 1;
        int w = (layer.w-1)/layer.stride + 1;
        int c = layer.c;
        image output = float_to_image(h,w,c,layer.output+b*h*w*c);
        image output_delta = float_to_image(h,w,c,layer.delta+b*h*w*c);

        int i,j,k;
        for(k = 0; k < input.c; ++k){
            for(i = 0; i < input.h; ++i){
                for(j = 0; j < input.w; ++j){
                    float val = get_pixel(input, i, j, k);
                    float cur = get_pixel(output, i/layer.stride, j/layer.stride, k);
                    float d = get_pixel(output_delta, i/layer.stride, j/layer.stride, k);
                    if(val == cur) {
                        set_pixel(input_delta, i, j, k, d);
                    }
                    else set_pixel(input_delta, i, j, k, 0);
                }
            }
        }
    }
}

