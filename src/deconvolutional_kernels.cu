#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"

void forward_deconvolutional_layer_gpu(layer l, network_state state)
{
    int i;
    int out_h = l.out_h;
    int out_w = l.out_w;
    int size = out_h*out_w;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights_gpu;
        float *b = state.input + i*l.c*l.h*l.w;
        float *c = state.workspace;

        gemm_ongpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_ongpu(c, l.n, out_h, out_w, l.size, l.stride, l.pad, l.output_gpu+i*l.n*size);
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array_ongpu(l.output_gpu, l.batch*l.n*size, l.activation);
}

void backward_deconvolutional_layer_gpu(layer l, network_state state)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int size = out_h*out_w;
    int i;

    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(state.delta) memset(state.delta, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = state.input + i*m*n;
        float *b = state.workspace;
        float *c = l.weight_updates_gpu;

        im2col_ongpu(l.delta_gpu + i*l.n*size, l.n, out_h, out_w, 
                l.size, l.stride, l.pad, b);
        gemm_ongpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(state.delta){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights_gpu;
            float *b = state.workspace;
            float *c = state.delta + i*n*m;

            gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void pull_deconvolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_deconvolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_deconvolutional_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_ongpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);

    if(l.scales_gpu){
        axpy_ongpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
        scal_ongpu(l.n, momentum, l.scale_updates_gpu, 1);
    }

    axpy_ongpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_ongpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_ongpu(size, momentum, l.weight_updates_gpu, 1);
}

