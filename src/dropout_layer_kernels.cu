extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) output[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

extern "C" void forward_dropout_layer_gpu(dropout_layer layer, float * input)
{
    int j;
    int size = layer.inputs*layer.batch;
    for(j = 0; j < size; ++j) layer.rand[j] = rand_uniform();
    cuda_push_array(layer.rand_gpu, layer.rand, layer.inputs*layer.batch);

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(input, size, layer.rand_gpu, layer.probability,
            layer.scale, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_dropout_layer_gpu(dropout_layer layer, float *delta)
{
    if(!delta) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(delta, size, layer.rand_gpu, layer.probability,
            layer.scale, delta);
    check_error(cudaPeekAtLastError());
}
