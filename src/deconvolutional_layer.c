#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>


static size_t get_workspace_size(layer l){
    return (size_t)l.h*l.w*l.size*l.size*l.c*sizeof(float);
}

int deconvolutional_out_height(layer l)
{
    return (l.h) * l.stride + l.size/2 - l.pad;
}

int deconvolutional_out_width(layer l)
{
    return (l.w) * l.stride + l.size/2 - l.pad;
}

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation, int batch_normalize)
{
    int i;
    layer l = {};
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.weights = (float*)calloc(c*n*size*size, sizeof(float));
    l.weight_updates = (float*)calloc(c*n*size*size, sizeof(float));

    l.biases = (float*)calloc(n, sizeof(float));
    l.bias_updates = (float*)calloc(n, sizeof(float));
    float scale = 1./sqrt((float)size*size*c);
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        l.biases[i] = scale;
    }
    l.pad = l.size/2;

    l.out_h = (l.h) * l.stride + l.size/2 - l.pad;
    l.out_w = (l.w) * l.stride + l.size/2 - l.pad;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (float*)calloc(l.batch*l.out_h * l.out_w * n, sizeof(float));
    l.delta  = (float*)calloc(l.batch*l.out_h * l.out_w * n, sizeof(float));

    l.forward = forward_deconvolutional_layer;
    l.backward = backward_deconvolutional_layer;
    l.update = update_deconvolutional_layer;

    l.batch_normalize = batch_normalize;

    if(batch_normalize){
        l.scales = (float*)calloc(n, sizeof(float));
        l.scale_updates = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float*)calloc(n, sizeof(float));
        l.variance = (float*)calloc(n, sizeof(float));

        l.mean_delta = (float*)calloc(n, sizeof(float));
        l.variance_delta = (float*)calloc(n, sizeof(float));

        l.rolling_mean = (float*)calloc(n, sizeof(float));
        l.rolling_variance = (float*)calloc(n, sizeof(float));
        l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_deconvolutional_layer_gpu;
    l.backward_gpu = backward_deconvolutional_layer_gpu;
    l.update_gpu = update_deconvolutional_layer_gpu;

    if(gpu_index >= 0){

        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);
        }
    }
    #ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
    #endif
#endif

    l.activation = activation;
    l.workspace_size = get_workspace_size(l);

    fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void resize_deconvolutional_layer(layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = (l->h) * l->stride + l->size/2 - l->pad;
    l->out_w = (l->w) * l->stride + l->size/2 - l->pad;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = (float*)realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = (float*)realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = (float*)realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
    #ifdef CUDNN
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
        cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    #endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void forward_deconvolutional_layer(const layer l, network_state state)
{
    int i;
    int out_h = l.out_h;
    int out_w = l.out_w;
    int size = out_h*out_w;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights;
        float *b = state.input + i*l.c*l.h*l.w;
        float *c = state.workspace;

        gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(c, l.n, out_h, out_w, l.size, l.stride, 0, l.output+i*l.n*size);
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }
    activate_array(l.output, l.batch*l.n*size, l.activation);
}

void backward_deconvolutional_layer(layer l, network_state state)
{
    float alpha = 1./l.batch;
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);
    int size = out_h*out_w;
    int i;

    gradient_array(l.output, size*l.n*l.batch, l.activation, l.delta);
    if(l.batch_normalize){
        backward_batchnorm_layer(l, state);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
    }

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = state.input + i*m*n;
        float *b = state.workspace;
        float *c = l.weight_updates;

        im2col_cpu(l.delta + i*l.n*size, l.n, out_h, out_w, 
                l.size, l.stride, 0, b);
        gemm(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(state.delta){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights;
            float *b = state.workspace;
            float *c = state.delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void update_deconvolutional_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}



