#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int deconvolutional_out_height(deconvolutional_layer layer)
{
    int h = layer.stride*(layer.h - 1) + layer.size;
    return h;
}

int deconvolutional_out_width(deconvolutional_layer layer)
{
    int w = layer.stride*(layer.w - 1) + layer.size;
    return w;
}

int deconvolutional_out_size(deconvolutional_layer layer)
{
    return deconvolutional_out_height(layer) * deconvolutional_out_width(layer);
}

image get_deconvolutional_image(deconvolutional_layer layer)
{
    int h,w,c;
    h = deconvolutional_out_height(layer);
    w = deconvolutional_out_width(layer);
    c = layer.n;
    return float_to_image(h,w,c,layer.output);
}

image get_deconvolutional_delta(deconvolutional_layer layer)
{
    int h,w,c;
    h = deconvolutional_out_height(layer);
    w = deconvolutional_out_width(layer);
    c = layer.n;
    return float_to_image(h,w,c,layer.delta);
}

deconvolutional_layer *make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation, float learning_rate, float momentum, float decay)
{
    int i;
    deconvolutional_layer *layer = calloc(1, sizeof(deconvolutional_layer));

    layer->learning_rate = learning_rate;
    layer->momentum = momentum;
    layer->decay = decay;

    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->batch = batch;
    layer->stride = stride;
    layer->size = size;

    layer->filters = calloc(c*n*size*size, sizeof(float));
    layer->filter_updates = calloc(c*n*size*size, sizeof(float));

    layer->biases = calloc(n, sizeof(float));
    layer->bias_updates = calloc(n, sizeof(float));
    float scale = 1./sqrt(size*size*c);
    for(i = 0; i < c*n*size*size; ++i) layer->filters[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        layer->biases[i] = scale;
    }
    int out_h = deconvolutional_out_height(*layer);
    int out_w = deconvolutional_out_width(*layer);

    layer->col_image = calloc(h*w*size*size*n, sizeof(float));
    layer->output = calloc(layer->batch*out_h * out_w * n, sizeof(float));
    layer->delta  = calloc(layer->batch*out_h * out_w * n, sizeof(float));

    #ifdef GPU
    layer->filters_gpu = cuda_make_array(layer->filters, c*n*size*size);
    layer->filter_updates_gpu = cuda_make_array(layer->filter_updates, c*n*size*size);

    layer->biases_gpu = cuda_make_array(layer->biases, n);
    layer->bias_updates_gpu = cuda_make_array(layer->bias_updates, n);

    layer->col_image_gpu = cuda_make_array(layer->col_image, h*w*size*size*n);
    layer->delta_gpu = cuda_make_array(layer->delta, layer->batch*out_h*out_w*n);
    layer->output_gpu = cuda_make_array(layer->output, layer->batch*out_h*out_w*n);
    #endif

    layer->activation = activation;

    fprintf(stderr, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return layer;
}

void resize_deconvolutional_layer(deconvolutional_layer *layer, int h, int w)
{
    layer->h = h;
    layer->w = w;
    int out_h = deconvolutional_out_height(*layer);
    int out_w = deconvolutional_out_width(*layer);

    layer->col_image = realloc(layer->col_image,
                                out_h*out_w*layer->size*layer->size*layer->c*sizeof(float));
    layer->output = realloc(layer->output,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));
    layer->delta  = realloc(layer->delta,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));
    #ifdef GPU
    cuda_free(layer->col_image_gpu);
    cuda_free(layer->delta_gpu);
    cuda_free(layer->output_gpu);

    layer->col_image_gpu = cuda_make_array(layer->col_image, out_h*out_w*layer->size*layer->size*layer->c);
    layer->delta_gpu = cuda_make_array(layer->delta, layer->batch*out_h*out_w*layer->n);
    layer->output_gpu = cuda_make_array(layer->output, layer->batch*out_h*out_w*layer->n);
    #endif
}

void forward_deconvolutional_layer(const deconvolutional_layer layer, float *in)
{
    int i;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;

    int m = layer.size*layer.size*layer.n;
    int n = layer.h*layer.w;
    int k = layer.c;

    bias_output(layer.output, layer.biases, layer.batch, layer.n, size);

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.filters;
        float *b = in + i*layer.c*layer.h*layer.w;
        float *c = layer.col_image;

        gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.output+i*layer.n*size);
    }
    activate_array(layer.output, layer.batch*layer.n*size, layer.activation);
}

void backward_deconvolutional_layer(deconvolutional_layer layer, float *in, float *delta)
{
    float alpha = 1./layer.batch;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;
    int i;

    gradient_array(layer.output, size*layer.n*layer.batch, layer.activation, layer.delta);
    backward_bias(layer.bias_updates, layer.delta, layer.batch, layer.n, size);

    if(delta) memset(delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));

    for(i = 0; i < layer.batch; ++i){
        int m = layer.c;
        int n = layer.size*layer.size*layer.n;
        int k = layer.h*layer.w;

        float *a = in + i*m*n;
        float *b = layer.col_image;
        float *c = layer.filter_updates;

        im2col_cpu(layer.delta + i*layer.n*size, layer.n, out_h, out_w, 
                layer.size, layer.stride, 0, b);
        gemm(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(delta){
            int m = layer.c;
            int n = layer.h*layer.w;
            int k = layer.size*layer.size*layer.n;

            float *a = layer.filters;
            float *b = layer.col_image;
            float *c = delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void update_deconvolutional_layer(deconvolutional_layer layer)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_cpu(layer.n, layer.learning_rate, layer.bias_updates, 1, layer.biases, 1);
    scal_cpu(layer.n, layer.momentum, layer.bias_updates, 1);

    axpy_cpu(size, -layer.decay, layer.filters, 1, layer.filter_updates, 1);
    axpy_cpu(size, layer.learning_rate, layer.filter_updates, 1, layer.filters, 1);
    scal_cpu(size, layer.momentum, layer.filter_updates, 1);
}



