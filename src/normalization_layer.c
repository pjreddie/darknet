#include "normalization_layer.h"
#include <stdio.h>

image get_normalization_image(normalization_layer layer)
{
    int h = layer.h;
    int w = layer.w;
    int c = layer.c;
    return float_to_image(h,w,c,layer.output);
}

image get_normalization_delta(normalization_layer layer)
{
    int h = layer.h;
    int w = layer.w;
    int c = layer.c;
    return float_to_image(h,w,c,layer.delta);
}

normalization_layer *make_normalization_layer(int batch, int h, int w, int c, int size, float alpha, float beta, float kappa)
{
    fprintf(stderr, "Local Response Normalization Layer: %d x %d x %d image, %d size\n", h,w,c,size);
    normalization_layer *layer = calloc(1, sizeof(normalization_layer));
    layer->batch = batch;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->kappa = kappa;
    layer->size = size;
    layer->alpha = alpha;
    layer->beta = beta;
    layer->output = calloc(h * w * c * batch, sizeof(float));
    layer->delta = calloc(h * w * c * batch, sizeof(float));
    layer->sums = calloc(h*w, sizeof(float));
    return layer;
}

void resize_normalization_layer(normalization_layer *layer, int h, int w, int c)
{
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->output = realloc(layer->output, h * w * c * layer->batch * sizeof(float));
    layer->delta = realloc(layer->delta, h * w * c * layer->batch * sizeof(float));
    layer->sums = realloc(layer->sums, h*w * sizeof(float));
}

void add_square_array(float *src, float *dest, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        dest[i] += src[i]*src[i];
    }
}
void sub_square_array(float *src, float *dest, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        dest[i] -= src[i]*src[i];
    }
}

void forward_normalization_layer(const normalization_layer layer, float *in)
{
    int i,j,k;
    memset(layer.sums, 0, layer.h*layer.w*sizeof(float));
    int imsize = layer.h*layer.w;
    for(j = 0; j < layer.size/2; ++j){
        if(j < layer.c) add_square_array(in+j*imsize, layer.sums, imsize);
    }
    for(k = 0; k < layer.c; ++k){
        int next = k+layer.size/2;
        int prev = k-layer.size/2-1;
        if(next < layer.c) add_square_array(in+next*imsize, layer.sums, imsize);
        if(prev > 0)       sub_square_array(in+prev*imsize, layer.sums, imsize);
        for(i = 0; i < imsize; ++i){
            layer.output[k*imsize + i] = in[k*imsize+i] / pow(layer.kappa + layer.alpha * layer.sums[i], layer.beta);
        }
    }
}

void backward_normalization_layer(const normalization_layer layer, float *in, float *delta)
{
    //TODO!
}

void visualize_normalization_layer(normalization_layer layer, char *window)
{
    image delta = get_normalization_image(layer);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    show_image(dc, buff);
    save_image(dc, buff);
    free_image(dc);
}
