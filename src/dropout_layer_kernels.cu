extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
#include "params.h"
}

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

extern "C" void forward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if (!state.train) return;
    int j;
    int size = layer.inputs*layer.batch;
    for(j = 0; j < size; ++j) layer.rand[j] = rand_uniform();
    cuda_push_array(layer.rand_gpu, layer.rand, layer.inputs*layer.batch);

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(state.input, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if(!state.delta) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(state.delta, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}
