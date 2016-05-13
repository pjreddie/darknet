#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] > 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += abs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_filters_kernel(float *filters, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += abs(filters[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (filters[f*size + i] > 0) ? mean : -mean;
    }
}

void binarize_filters_gpu(float *filters, int n, int size, float *binary)
{
    binarize_filters_kernel<<<cuda_gridsize(n), BLOCK>>>(filters, n, size, binary);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = convolutional_out_height(l)*
        convolutional_out_width(l);

    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_filters_gpu(l.filters_gpu, l.n, l.c*l.size*l.size, l.binary_filters_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_filters_gpu(l.filters_gpu, l.n, l.c*l.size*l.size, l.binary_filters_gpu);
        swap_binary(&l);
        for(i = 0; i < l.batch; ++i){
            binarize_input_gpu(state.input + i*l.inputs, l.c, l.h*l.w, l.binary_input_gpu + i*l.inputs);
        }
        state.input = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                state.input,
                l.filterDesc,
                l.filters_gpu,
                l.convDesc,
                l.fw_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    for(i = 0; i < l.batch; ++i){
        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        float * a = l.filters_gpu;
        float * b = state.workspace;
        float * c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, n);

    activate_array_ongpu(l.output_gpu, m*n*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array_ongpu(l.output_gpu, m*k*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, k);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
    }

    if(l.xnor) state.input = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            state.input,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dfilterDesc,
            l.filter_updates_gpu);

    if(state.delta){
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.filterDesc,
                l.filters_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                state.delta);
    }

#else
    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = state.workspace;
        float * c = l.filter_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            if(l.binary || l.xnor) swap_binary(&l);
            float * a = l.filters_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) swap_binary(&l);
        }
    }
#endif
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);

    axpy_ongpu(size, -decay*batch, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, learning_rate/batch, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, momentum, layer.filter_updates_gpu, 1);
}


