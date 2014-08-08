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

float get_max_region(image im, int h, int w, int c, int size)
{
    int i,j;
    int lower = (-size-1)/2 + 1;
    int upper = size/2 + 1;
    
    int lh = (h-lower < 0)      ? 0 : h-lower;
    int uh = (h+upper > im.h)   ? im.h : h+upper;

    int lw = (w-lower < 0)      ? 0 : w-lower;
    int uw = (w+upper > im.w)   ? im.w : w+upper;
    
    //printf("%d\n", -3/2);
    //printf("%d %d\n", lower, upper);
    //printf("%d %d %d %d\n", lh, uh, lw, uw);
    
    float max = -FLT_MAX;
    for(i = lh; i < uh; ++i){
        for(j = lw; j < uw; ++j){
            float val = get_pixel(im, i, j, c);
            if (val > max) max = val;
        }
    }
    return max;
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
        for(k = 0; k < input.c; ++k){
            for(i = 0; i < input.h; i += layer.stride){
                for(j = 0; j < input.w; j += layer.stride){
                    float max = get_max_region(input, i, j, k, layer.size);
                    set_pixel(output, i/layer.stride, j/layer.stride, k, max);
                }
            }
        }
    }
}

float set_max_region_delta(image im, image delta, int h, int w, int c, int size, float max, float error)
{
    int i,j;
    int lower = (-size-1)/2 + 1;
    int upper = size/2 + 1;
    
    int lh = (h-lower < 0)      ? 0 : h-lower;
    int uh = (h+upper > im.h)   ? im.h : h+upper;

    int lw = (w-lower < 0)      ? 0 : w-lower;
    int uw = (w+upper > im.w)   ? im.w : w+upper;
    
    for(i = lh; i < uh; ++i){
        for(j = lw; j < uw; ++j){
            float val = get_pixel(im, i, j, c);
            if (val == max){
               add_pixel(delta, i, j, c, error);
            }
        }
    }
    return max;
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
        zero_image(input_delta);

        int i,j,k;
        for(k = 0; k < input.c; ++k){
            for(i = 0; i < input.h; i += layer.stride){
                for(j = 0; j < input.w; j += layer.stride){
                    float max = get_pixel(output, i/layer.stride, j/layer.stride, k);
                    float error = get_pixel(output_delta, i/layer.stride, j/layer.stride, k);
                    set_max_region_delta(input, input_delta, i, j, k, layer.size, max, error);
                }
            }
        }
    }
}

