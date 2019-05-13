#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include "dropout_layer.h"
#include "dark_cuda.h"
#include "utils.h"

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if (!state.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>>(state.input, size, layer.rand_gpu, layer.probability, layer.scale);
    CHECK_CUDA(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if(!state.delta) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>>(state.delta, size, layer.rand_gpu, layer.probability, layer.scale);
    CHECK_CUDA(cudaPeekAtLastError());
}
