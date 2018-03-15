#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

extern "C" void forward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights_gpu;
        float *b = net.input_gpu + i*l.c*l.h*l.w;
        float *c = net.workspace;

        gemm_gpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_gpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output_gpu+i*l.outputs);
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array_gpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

extern "C" void backward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = net.input_gpu + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates_gpu;

        im2col_gpu(l.delta_gpu + i*l.outputs, l.out_c, l.out_h, l.out_w, 
                l.size, l.stride, l.pad, b);
        gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta_gpu){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights_gpu;
            float *b = net.workspace;
            float *c = net.delta_gpu + i*n*m;

            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

extern "C" void pull_deconvolutional_layer(layer l)
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

extern "C" void push_deconvolutional_layer(layer l)
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

void update_deconvolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, size, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(size, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}

