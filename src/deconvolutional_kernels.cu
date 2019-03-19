#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "dark_cuda.h"

extern "C" void forward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state)
{
    int i;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;

    int m = layer.size*layer.size*layer.n;
    int n = layer.h*layer.w;
    int k = layer.c;

    fill_ongpu(layer.outputs*layer.batch, 0, layer.output_gpu, 1);

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.weights_gpu;
        float *b = state.input + i*layer.c*layer.h*layer.w;
        float *c = layer.col_image_gpu;

        gemm_ongpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_ongpu(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.output_gpu+i*layer.n*size);
    }
    add_bias_gpu(layer.output_gpu, layer.biases_gpu, layer.batch, layer.n, size);
    activate_array(layer.output_gpu, layer.batch*layer.n*size, layer.activation);
}

extern "C" void backward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state)
{
    float alpha = 1./layer.batch;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;
    int i;

    gradient_array(layer.output_gpu, size*layer.n*layer.batch, layer.activation, layer.delta_gpu);
    backward_bias(layer.bias_updates_gpu, layer.delta, layer.batch, layer.n, size);

    if(state.delta) memset(state.delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));

    for(i = 0; i < layer.batch; ++i){
        int m = layer.c;
        int n = layer.size*layer.size*layer.n;
        int k = layer.h*layer.w;

        float *a = state.input + i*m*n;
        float *b = layer.col_image_gpu;
        float *c = layer.weight_updates_gpu;

        im2col_ongpu(layer.delta_gpu + i*layer.n*size, layer.n, out_h, out_w,
                layer.size, layer.stride, 0, b);
        gemm_ongpu(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(state.delta){
            int m = layer.c;
            int n = layer.h*layer.w;
            int k = layer.size*layer.size*layer.n;

            float *a = layer.weights_gpu;
            float *b = layer.col_image_gpu;
            float *c = state.delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

extern "C" void pull_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

extern "C" void push_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

extern "C" void update_deconvolutional_layer_gpu(deconvolutional_layer layer, int skip, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(size, -decay, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
    axpy_ongpu(size, learning_rate, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
    scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);
}
