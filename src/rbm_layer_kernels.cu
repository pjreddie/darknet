#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "rbm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"
#include "connected_layer.h"

#include <stdio.h>
}

void activate_hidden1_gpu(rbm_layer l, network_state state) {
    //firstly copy biases to the output
    int i;
    for(i = 0; i < l.batch; ++i){
        //copy_ongpu_offset(l.outputs, l.biases_gpu, 0, 1, l.output_gpu, i*l.outputs, 1);
        copy_ongpu(l.outputs, l.biases_gpu, 1, l.output_gpu + i*l.outputs, 1);
    }
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = state.input; 
    float *c = l.output_gpu;
    float *b = l.weights_gpu;
    //secondly add input*weight to the output
    gemm_ongpu(0,1,m,n,k,1,a,k,b,k,1,c,n); // in gemm
    //thirdly activate the output
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation); // in activation
}
void activate_hidden2_gpu(rbm_layer l) {
    //firstly copy biases to the houtput
    int i;
    for(i = 0; i < l.batch; ++i){
        copy_ongpu(l.outputs, l.biases_gpu, 1, l.houtput_gpu + i*l.outputs, 1); // in blas
    }
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = l.vinput_gpu; 
    float *c = l.houtput_gpu;
    float *b = l.weights_gpu;
    //secondly add input*weight to the houtput
    gemm_ongpu(0,1,m,n,k,1,a,k,b,k,1,c,n); // in gemm
    //thirdly activate the houtput
    activate_array_ongpu(l.houtput_gpu, l.outputs*l.batch, l.activation); // in activation
}
void activate_visible_gpu(rbm_layer l) {
    //firstly copy vbiases to the voutput
    int i;
    for(i = 0; i < l.batch; ++i) {
        copy_ongpu(l.inputs, l.vbiases_gpu, 1, l.vinput_gpu + i*l.inputs, 1);
    }
    //secondly add output*weight to the voutput
    int m = l.batch;
    int k = l.outputs;
    int n = l.inputs;
    float* a = l.output_gpu;
/*
    for(i=0; i<l.batch*l.outputs; ++i){
        a[i] = (rand_uniform<a[i])? 1.0 : 0;
    }
*/
    float* b = l.weights_gpu;
    float* c = l.vinput_gpu;
    gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n); // in gemm
    //thirdly activate the output
    activate_array_ongpu(l.vinput_gpu, l.inputs*l.batch, l.activation); // in activation
}
void preforward_rbm_layer_gpu(rbm_layer l, network_state state) {
    activate_hidden1_gpu(l, state);
    activate_visible_gpu(l);
    activate_hidden2_gpu(l);
}

__global__ void rbm_axpy_kernel(int N, float ALPHA, float BETAX, float BETAY, float *X, int INCX, float *Y, int INCY, float *Z, int INCZ) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Z[i*INCZ] += ALPHA*(BETAX*X[i*INCX]+BETAY*Y[i*INCY]);
}
void rbm_axpy_gpu(int N, float ALPHA, float BETAX, float BETAY, float *X, int INCX, float *Y, int INCY, float *Z, int INCZ) {
    rbm_axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, BETAX, BETAY, X, INCX, Y, INCY, Z, INCZ);
    //printf("rbm_axpy_kernel.\n");
    check_error(cudaPeekAtLastError());
}

__global__ void rbm_gemm_kernel(int outputs, int inputs, float* h1, float* h2, float* v1, float* v2, float* weights) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= outputs) return;
    int j;
    for(j=0; j<inputs; j++) {
        weights[i*inputs+j] += h1[i]*v1[j] - h2[i]*v2[j];
    }
    
}
void rbm_gemm_gpu(int outputs, int inputs, float* h1, float* h2, float* v1, float* v2, float* weights) {
    rbm_gemm_kernel<<<cuda_gridsize(outputs), BLOCK>>>(outputs, inputs, h1, h2, v1, v2, weights);
    //printf("rbm_gemm_gpu\n");
    check_error(cudaPeekAtLastError());
}

void prebackward_rbm_layer_gpu(rbm_layer l, network_state state) {
    //compute updates
    int i;
    for(i=0; i<l.batch; i++) {
        rbm_axpy_gpu(l.outputs, 1, 1, -1, l.output_gpu + i*l.outputs, 1, l.houtput_gpu + i*l.outputs, 1, l.bias_updates_gpu, 1); 
        rbm_axpy_gpu(l.inputs, 1, 1, -1, state.input + i*l.inputs, 1, l.vinput_gpu + i*l.inputs, 1, l.vbias_updates_gpu, 1);
        rbm_gemm_gpu(l.outputs, l.inputs, l.output_gpu + i*l.outputs, l.houtput_gpu + i*l.outputs, 
					 state.input + i*l.inputs, l.vinput_gpu + i*l.inputs, l.weight_updates_gpu);
    }
}

void preupdate_rbm_layer_gpu(rbm_layer l, int batch, float learning_rate, float momentum, float decay) {
    // all in blas
    axpy_ongpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.outputs, momentum, l.bias_updates_gpu, 1);

    axpy_ongpu(l.inputs, learning_rate/batch, l.vbias_updates_gpu, 1, l.vbiases_gpu, 1);
    scal_ongpu(l.inputs, momentum, l.vbias_updates_gpu, 1);

    axpy_ongpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_ongpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_ongpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
}

void forward_rbm_layer_gpu(rbm_layer l, network_state state)
{
    l.type = CONNECTED;
    forward_connected_layer_gpu(l, state);
    l.type = RBM;
}

void pull_rbm_layer(rbm_layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);

    cuda_pull_array(l.vbiases_gpu, l.vbiases, l.inputs);
    cuda_pull_array(l.vbias_updates_gpu, l.vbias_updates, l.inputs);

    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_rbm_layer(rbm_layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);

    cuda_push_array(l.vbiases_gpu, l.vbiases, l.inputs);
    cuda_push_array(l.vbias_updates_gpu, l.vbias_updates, l.inputs);

    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}
