#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"


#include "softmax_layer.h"
#include "cuda.h"
#include "blas.h"


__global__ void forward_softmax_layer_kernel(int n, int batch, float *input, float temp, float *output)
{
    int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(b >= batch) return;

    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i+b*n];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        sum += exp(input[i+b*n]/temp-largest/temp);
    }
    sum = (sum != 0) ? largest/temp+log(sum) : largest-100;
    for(i = 0; i < n; ++i){
        output[i+b*n] = exp(input[i+b*n]/temp-sum);
    }
}

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer layer, network_state state)
{
    int inputs = layer.inputs / layer.groups;
    int batch = layer.batch * layer.groups;
    forward_softmax_layer_kernel<<<cuda_gridsize(batch), BLOCK>>>(inputs, batch, state.input, layer.temperature, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_softmax_layer_gpu(const softmax_layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
}

/* This is if you want softmax w/o log-loss classification. You probably don't.
   int i,j,b;
   for(b = 0; b < layer.batch; ++b){
   for(i = 0; i < layer.inputs; ++i){
   for(j = 0; j < layer.inputs; ++j){
   int d = (i==j);
   layer.jacobian[b*layer.inputs*layer.inputs + i*layer.inputs + j] = 
   layer.output[b*layer.inputs + i] * (d - layer.output[b*layer.inputs + j]);
   }
   }
   }
   for(b = 0; b < layer.batch; ++b){
   int M = layer.inputs;
   int N = 1;
   int K = layer.inputs;
   float *A = layer.jacobian + b*layer.inputs*layer.inputs;
   float *B = layer.delta + b*layer.inputs;
   float *C = delta + b*layer.inputs;
   gemm(0,0,M,N,K,1,A,K,B,N,0,C,N);
   }
 */
