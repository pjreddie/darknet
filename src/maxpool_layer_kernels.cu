#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include "maxpool_layer.h"
#include "convolutional_layer.h"
#include "blas.h"
#include "dark_cuda.h"

__global__ void forward_maxpool_depth_layer_kernel(int n, int w, int h, int c, int out_c, int batch, float *input, float *output, int *indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int j = id % w;
    id = id / w;
    int i = id % h;
    id = id / h;
    //int g = id % out_c;
    //id = id / out_c;
    int b = id % batch;

    int k;
    for (int g = 0; g < out_c; ++g)
    {
        int out_index = j + w*(i + h*(g + out_c*b));
        float max = -FLT_MAX;
        int max_i = -1;

        for (k = g; k < c; k += out_c)
        {
            int in_index = j + w*(i + h*(k + c*b));
            float val = input[in_index];

            max_i = (val > max) ? in_index : max_i;
            max = (val > max) ? val : max;
        }
        output[out_index] = max;
        if (indexes) indexes[out_index] = max_i;
    }
}


__global__ void backward_maxpool_depth_layer_kernel(int n, int w, int h, int c, int batch, float *delta, float *prev_delta, int *indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int index = indexes[id];
    prev_delta[index] += delta[id];
}


__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *input, float *output, int *indexes)
{
    int h = (in_h + pad - size) / stride_y + 1;
    int w = (in_w + pad - size) / stride_x + 1;
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

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride_y + l;
            int cur_w = w_offset + j*stride_x + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    if (indexes) indexes[out_index] = max_i;
}

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *delta, float *prev_delta, int *indexes)
{
    int h = (in_h + pad - size) / stride_y + 1;
    int w = (in_w + pad - size) / stride_x + 1;
    int c = in_c;
    int area_x = (size - 1) / stride_x;
    int area_y = (size - 1) / stride_y;

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

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    float d = 0;
    int l, m;
    for(l = -area_y; l < area_y+1; ++l){
        for(m = -area_x; m < area_x+1; ++m){
            int out_w = (j-w_offset)/stride_x + m;
            int out_h = (i-h_offset)/stride_y + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}

extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network_state state)
{
    if (layer.maxpool_depth) {
        int h = layer.out_h;
        int w = layer.out_w;
        int c = 1;// layer.out_c;

        size_t n = h*w*c*layer.batch;

        forward_maxpool_depth_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(
            n, layer.w, layer.h, layer.c, layer.out_c, layer.batch, state.input, layer.output_gpu, layer.indexes_gpu);
        CHECK_CUDA(cudaPeekAtLastError());

        return;
    }

#ifdef CUDNN_DISABLED
    if (!state.train && layer.stride == layer.size) {
        // cudnnPoolingBackward
        cudnnStatus_t maxpool_status;

        float alpha = 1, beta = 0;
        maxpool_status = cudnnPoolingForward(
            cudnn_handle(),
            layer.poolingDesc,
            &alpha,
            layer.srcTensorDesc,
            state.input,
            &beta,
            layer.dstTensorDesc,
            layer.output_gpu);

        //maxpool_status = cudnnDestroyPoolingDescriptor(poolingDesc);
        //cudnnDestroyTensorDescriptor(layer.srcTensorDesc);
        //cudnnDestroyTensorDescriptor(layer.dstTensorDesc);

    }
    else
#endif
    {
        int h = layer.out_h;
        int w = layer.out_w;
        int c = layer.out_c;

        size_t n = h*w*c*layer.batch;

        forward_maxpool_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (n, layer.h, layer.w, layer.c, layer.stride_x, layer.stride_y, layer.size, layer.pad, state.input, layer.output_gpu, layer.indexes_gpu);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    if (layer.antialiasing) {
        network_state s = { 0 };
        s.train = state.train;
        s.workspace = state.workspace;
        s.net = state.net;
        if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
        s.input = layer.output_gpu;
        forward_convolutional_layer_gpu(*(layer.input_layer), s);
        simple_copy_ongpu(layer.outputs*layer.batch, layer.output_gpu, layer.input_antialiasing_gpu);
        simple_copy_ongpu(layer.input_layer->outputs*layer.input_layer->batch, layer.input_layer->output_gpu, layer.output_gpu);
    }
}

extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network_state state)
{
    if (layer.antialiasing) {
        network_state s = { 0 };
        s.train = state.train;
        s.workspace = state.workspace;
        s.net = state.net;
        s.delta = layer.delta_gpu;  // s.delta will be returned to l.delta_gpu
        s.input = layer.input_antialiasing_gpu;
        //if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
        simple_copy_ongpu(layer.input_layer->outputs*layer.input_layer->batch, layer.delta_gpu, layer.input_layer->delta_gpu);
        backward_convolutional_layer_gpu(*(layer.input_layer), s);

        //simple_copy_ongpu(layer.outputs*layer.batch, layer.input_antialiasing_gpu, layer.output_gpu);
    }

    if (layer.maxpool_depth) {
        int h = layer.out_h;
        int w = layer.out_w;
        int c = layer.out_c;

        size_t n = h * w * c * layer.batch;

        backward_maxpool_depth_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(n, layer.w, layer.h, layer.c, layer.batch, layer.delta_gpu, state.delta, layer.indexes_gpu);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    size_t n = layer.h*layer.w*layer.c*layer.batch;

    backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(n, layer.h, layer.w, layer.c, layer.stride_x, layer.stride_y, layer.size, layer.pad, layer.delta_gpu, state.delta, layer.indexes_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}
