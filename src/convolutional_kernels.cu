extern "C" {
#include "convolutional_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void bias_output_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] = biases[filter];
}

void bias_output_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    bias_output_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if(p == 0){
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer layer, network_state state)
{
    int i;
    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = convolutional_out_height(layer)*
        convolutional_out_width(layer);

    bias_output_gpu(layer.output_gpu, layer.biases_gpu, layer.batch, layer.n, n);
    for(i = 0; i < layer.batch; ++i){
        im2col_ongpu(state.input + i*layer.c*layer.h*layer.w, layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, layer.pad, layer.col_image_gpu);
        float * a = layer.filters_gpu;
        float * b = layer.col_image_gpu;
        float * c = layer.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }
    activate_array_ongpu(layer.output_gpu, m*n*layer.batch, layer.activation);
}

void backward_convolutional_layer_gpu(convolutional_layer layer, network_state state)
{
    int i;
    int m = layer.n;
    int n = layer.size*layer.size*layer.c;
    int k = convolutional_out_height(layer)*
        convolutional_out_width(layer);

    gradient_array_ongpu(layer.output_gpu, m*k*layer.batch, layer.activation, layer.delta_gpu);
    backward_bias_gpu(layer.bias_updates_gpu, layer.delta_gpu, layer.batch, layer.n, k);

    for(i = 0; i < layer.batch; ++i){
        float * a = layer.delta_gpu;
        float * b = layer.col_image_gpu;
        float * c = layer.filter_updates_gpu;

        im2col_ongpu(state.input + i*layer.c*layer.h*layer.w, layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, layer.pad, layer.col_image_gpu);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){

            float * a = layer.filters_gpu;
            float * b = layer.delta_gpu;
            float * c = layer.col_image_gpu;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(layer.col_image_gpu, layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, layer.pad, state.delta + i*layer.c*layer.h*layer.w);
        }
    }
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(size, -decay*batch, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, learning_rate/batch, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, momentum, layer.filter_updates_gpu, 1);
}

