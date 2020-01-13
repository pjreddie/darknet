#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int deconvolutional_out_height(deconvolutional_layer l)
{
    int h = l.stride*(l.h - 1) + l.size;
    return h;
}

int deconvolutional_out_width(deconvolutional_layer l)
{
    int w = l.stride*(l.w - 1) + l.size;
    return w;
}

int deconvolutional_out_size(deconvolutional_layer l)
{
    return deconvolutional_out_height(l) * deconvolutional_out_width(l);
}

image get_deconvolutional_image(deconvolutional_layer l)
{
    int h,w,c;
    h = deconvolutional_out_height(l);
    w = deconvolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}

image get_deconvolutional_delta(deconvolutional_layer l)
{
    int h,w,c;
    h = deconvolutional_out_height(l);
    w = deconvolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}

deconvolutional_layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    deconvolutional_layer l = { (LAYER_TYPE)0 };
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.weights = (float*)xcalloc(c * n * size * size, sizeof(float));
    l.weight_updates = (float*)xcalloc(c * n * size * size, sizeof(float));

    l.biases = (float*)xcalloc(n, sizeof(float));
    l.bias_updates = (float*)xcalloc(n, sizeof(float));
    float scale = 1./sqrt(size*size*c);
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        l.biases[i] = scale;
    }
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);

    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.col_image = (float*)xcalloc(h * w * size * size * n, sizeof(float));
    l.output = (float*)xcalloc(l.batch * out_h * out_w * n, sizeof(float));
    l.delta = (float*)xcalloc(l.batch * out_h * out_w * n, sizeof(float));

    l.forward = forward_deconvolutional_layer;
    l.backward = backward_deconvolutional_layer;
    l.update = update_deconvolutional_layer;

    #ifdef GPU
    l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.col_image_gpu = cuda_make_array(l.col_image, h*w*size*size*n);
    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    #endif

    l.activation = activation;

    fprintf(stderr, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

void resize_deconvolutional_layer(deconvolutional_layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    int out_h = deconvolutional_out_height(*l);
    int out_w = deconvolutional_out_width(*l);

    l->col_image = (float*)xrealloc(l->col_image,
                                out_h*out_w*l->size*l->size*l->c*sizeof(float));
    l->output = (float*)xrealloc(l->output,
                                l->batch*out_h * out_w * l->n*sizeof(float));
    l->delta = (float*)xrealloc(l->delta,
                                l->batch*out_h * out_w * l->n*sizeof(float));
    #ifdef GPU
    cuda_free(l->col_image_gpu);
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->col_image_gpu = cuda_make_array(l->col_image, out_h*out_w*l->size*l->size*l->c);
    l->delta_gpu = cuda_make_array(l->delta, l->batch*out_h*out_w*l->n);
    l->output_gpu = cuda_make_array(l->output, l->batch*out_h*out_w*l->n);
    #endif
}

void forward_deconvolutional_layer(const deconvolutional_layer l, network_state state)
{
    int i;
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);
    int size = out_h*out_w;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights;
        float *b = state.input + i*l.c*l.h*l.w;
        float *c = l.col_image;

        gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(c, l.n, out_h, out_w, l.size, l.stride, 0, l.output+i*l.n*size);
    }
    add_bias(l.output, l.biases, l.batch, l.n, size);
    activate_array(l.output, l.batch*l.n*size, l.activation);
}

void backward_deconvolutional_layer(deconvolutional_layer l, network_state state)
{
    float alpha = 1./l.batch;
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);
    int size = out_h*out_w;
    int i;

    gradient_array(l.output, size*l.n*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, size);

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = state.input + i*m*n;
        float *b = l.col_image;
        float *c = l.weight_updates;

        im2col_cpu(l.delta + i*l.n*size, l.n, out_h, out_w,
                l.size, l.stride, 0, b);
        gemm(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(state.delta){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights;
            float *b = l.col_image;
            float *c = state.delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void update_deconvolutional_layer(deconvolutional_layer l, int skip, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    axpy_cpu(size, -decay, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}
