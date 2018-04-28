#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "maxpool_layer.h"
#include "cuda.h"
}

__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output, int *indexes)
{
    int h = (in_h + 2*pad)/stride;
    int w = (in_w + 2*pad)/stride;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad;
    int h_offset = -pad;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *delta, float *prev_delta, int *indexes)
{
    int h = (in_h + 2*pad)/stride;
    int w = (in_w + 2*pad)/stride;
    int c = in_c;
    int area = (size-1)/stride;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad;
    int h_offset = -pad;

    float d = 0;
    int l, m;
    for(l = -area; l < area+1; ++l){
        for(m = -area; m < area+1; ++m){
            int out_w = (j-w_offset)/stride + m;
            int out_h = (i-h_offset)/stride + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}

extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = h*w*c*layer.batch;

    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, net.input_gpu, layer.output_gpu, layer.indexes_gpu);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;

    backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, layer.delta_gpu, net.delta_gpu, layer.indexes_gpu);
    check_error(cudaPeekAtLastError());
}

