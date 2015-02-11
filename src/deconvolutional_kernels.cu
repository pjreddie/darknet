extern "C" {
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

extern "C" void forward_deconvolutional_layer_gpu(deconvolutional_layer layer, float *in)
{
    int i;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;

    int m = layer.size*layer.size*layer.n;
    int n = layer.h*layer.w;
    int k = layer.c;

    bias_output_gpu(layer.output_gpu, layer.biases_gpu, layer.batch, layer.n, size);

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.filters_gpu;
        float *b = in + i*layer.c*layer.h*layer.w;
        float *c = layer.col_image_gpu;

        gemm_ongpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_ongpu(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.output_gpu+i*layer.n*size);
    }
    activate_array(layer.output_gpu, layer.batch*layer.n*size, layer.activation);
}

extern "C" void backward_deconvolutional_layer_gpu(deconvolutional_layer layer, float *in, float *delta_gpu)
{
    float alpha = 1./layer.batch;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;
    int i;

    gradient_array(layer.output_gpu, size*layer.n*layer.batch, layer.activation, layer.delta_gpu);
    backward_bias(layer.bias_updates_gpu, layer.delta, layer.batch, layer.n, size);

    if(delta_gpu) memset(delta_gpu, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));

    for(i = 0; i < layer.batch; ++i){
        int m = layer.c;
        int n = layer.size*layer.size*layer.n;
        int k = layer.h*layer.w;

        float *a = in + i*m*n;
        float *b = layer.col_image_gpu;
        float *c = layer.filter_updates_gpu;

        im2col_ongpu(layer.delta_gpu + i*layer.n*size, layer.n, out_h, out_w, 
                layer.size, layer.stride, 0, b);
        gemm_ongpu(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(delta_gpu){
            int m = layer.c;
            int n = layer.h*layer.w;
            int k = layer.size*layer.size*layer.n;

            float *a = layer.filters_gpu;
            float *b = layer.col_image_gpu;
            float *c = delta_gpu + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

extern "C" void pull_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

extern "C" void push_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

extern "C" void update_deconvolutional_layer_gpu(deconvolutional_layer layer)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, layer.learning_rate, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n,layer.momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(size, -layer.decay, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, layer.learning_rate, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, layer.momentum, layer.filter_updates_gpu, 1);
}

