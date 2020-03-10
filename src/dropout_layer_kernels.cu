#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cstring>

#include "dropout_layer.h"
#include "dark_cuda.h"
#include "utils.h"
#include "blas.h"


__global__ void dropblock_fast_kernel(float *rand, float prob, int w, int h, int spatial, int filters, int block_size, float *drop_blocks_scale, float *output)
{
    const int threads = BLOCK;
    const int id = threadIdx.x;
    const int f = blockIdx.x % filters;
    const int b = blockIdx.x / filters;

    __shared__ int prob_block;
    __shared__ int index_block;

    if (id == 0) {
        prob_block = 1.0 * 1000000;
        index_block = -1;
    }
    __syncthreads();

    int i;
    for (i = id; i < spatial; i += threads) {
        int index = b*spatial*f + f*spatial + i;

        if (rand[index] < prob) {
            //Chose with the lowest rand[i]
            int new_val = rand[index] * 1000000;
            int old_val = atomicMin(&prob_block, new_val);
            if (new_val < old_val) {
                index_block = i;
                //if (b == 0) printf("\n rand[i] = %f, prob = %f, b = %d, f = %d, i = %d, index_block = %d \n", rand[i], prob, b, f, i, index_block);
            }
        }

    }
    __syncthreads();
    if (index_block == -1) return;


    int b_x = index_block % w;
    int b_y = index_block / w;

    b_x = max(0, min(b_x, w - block_size));
    b_y = max(0, min(b_y, h - block_size));

    int block_square_size = block_size * block_size;

    for (i = id; i < block_square_size; i += threads)
    {
        int i_x = i % w;
        int i_y = i / w;

        int x = b_x + i_x;
        int y = b_y + i_y;

        if (x < w && y < h) {
            int index = b*spatial*f + f*spatial + y*w + x;

            output[index] = 0;
        }
    }

    if (id == 0 && drop_blocks_scale) {
        atomicAdd(&drop_blocks_scale[b], 1);
        //if(b == 0) printf("\n index_block = %d \n", index_block);
    }

}

__global__ void set_scales_dropblock_kernel(float *drop_blocks_scale, int block_size_w, int block_size_h, int outputs, int batch)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= batch) return;

    const float prob = drop_blocks_scale[index] * block_size_w * block_size_h / (float)outputs;
    const float scale = 1.0f / (1.0f - prob);
    drop_blocks_scale[index] = scale;
}

__global__ void scale_dropblock_kernel(float *output, int size, int outputs, float *drop_blocks_scale)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= size) return;

    const int b = index / outputs;
    output[index] *= drop_blocks_scale[b];
}


__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}


void forward_dropout_layer_gpu(dropout_layer l, network_state state)
{
    if (!state.train) return;
    int iteration_num = get_current_iteration(state.net); // (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    //if (iteration_num < state.net.burn_in) return;

    // We gradually increase the block size and the probability of dropout - during the first half of the training
    float multiplier = 1.0;
    if(iteration_num < (state.net.max_batches*0.85))
        multiplier = (iteration_num / (float)(state.net.max_batches*0.85));

    // dropblock
    if (l.dropblock) {
        //l.probability = 1 / keep_prob
        //const int max_blocks_per_channel = 10;
        const float cur_prob = l.probability * multiplier;
        const float cur_scale = 1.f / (1.f - cur_prob);

        int block_width = l.dropblock_size_abs *multiplier;
        int block_height = l.dropblock_size_abs *multiplier;

        if (l.dropblock_size_rel) {
            block_width = l.dropblock_size_rel * l.w * multiplier;
            block_height = l.dropblock_size_rel * l.h * multiplier;
        }

        block_width = max_val_cmp(1, block_width);
        block_height = max_val_cmp(1, block_height);

        block_width = min_val_cmp(l.w, block_width);
        block_height = min_val_cmp(l.h, block_height);

        const int block_size = min_val_cmp(block_width, block_height);
        const float block_prob = cur_prob / (block_size*block_size);

        int size = l.inputs*l.batch;
        cuda_random(l.rand_gpu, size);

        fill_ongpu(l.batch, 0, l.drop_blocks_scale_gpu, 1);

        int num_blocks = l.batch * l.c;
        dropblock_fast_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (l.rand_gpu, block_prob, l.w, l.h, l.w*l.h, l.c, block_size, l.drop_blocks_scale_gpu, state.input);
        CHECK_CUDA(cudaPeekAtLastError());

        num_blocks = get_number_of_blocks(l.batch, BLOCK);
        set_scales_dropblock_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (l.drop_blocks_scale_gpu, block_size, block_size, l.outputs, l.batch);
        CHECK_CUDA(cudaPeekAtLastError());

        /*
        cuda_pull_array(l.drop_blocks_scale_gpu, l.drop_blocks_scale, l.batch);

        for (int b = 0; b < l.batch; ++b) {
            const float prob = l.drop_blocks_scale[b] * block_size * block_size / (float)l.outputs;
            const float scale = 1.0f / (1.0f - prob);
            //printf(" %d x %d - block_size = %d, block_size*block_size = %d , ", l.w, l.h, block_size, block_size*block_size);
            //printf(" , l.drop_blocks_scale[b] = %f, prob = %f, calc scale = %f \t cur_prob = %f, cur_scale = %f \n",
            //    l.drop_blocks_scale[b], prob, scale, cur_prob, cur_scale);
            l.drop_blocks_scale[b] = scale;
        }

        cuda_push_array(l.drop_blocks_scale_gpu, l.drop_blocks_scale, l.batch);
        */

        num_blocks = get_number_of_blocks(l.outputs * l.batch, BLOCK);
        scale_dropblock_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (state.input, l.outputs * l.batch, l.outputs, l.drop_blocks_scale_gpu);
        //scal_ongpu(l.outputs * l.batch, cur_scale, state.input, 1);
        //scal_ongpu(l.outputs * l.batch, l.drop_blocks_scale[0], state.input, 1);
        CHECK_CUDA(cudaPeekAtLastError());

    }
    // dropout
    else {
        int size = l.inputs*l.batch;
        cuda_random(l.rand_gpu, size);
        /*
        int i;
        for(i = 0; i < size; ++i){
            layer.rand[i] = rand_uniform();
        }
        cuda_push_array(layer.rand_gpu, layer.rand, size);
        */

        yoloswag420blazeit360noscope << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (state.input, size, l.rand_gpu, l.probability, l.scale);
        CHECK_CUDA(cudaPeekAtLastError());
    }
}

void backward_dropout_layer_gpu(dropout_layer l, network_state state)
{
    if(!state.delta) return;
    //int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);
    //if (iteration_num < state.net.burn_in) return;

    int size = l.inputs*l.batch;

    // dropblock
    if (l.dropblock) {
        int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);
        float multiplier = 1.0;
        if (iteration_num < (state.net.max_batches*0.85))
            multiplier = (iteration_num / (float)(state.net.max_batches*0.85));

        const float cur_prob = l.probability * multiplier;
        const float cur_scale = 1.f / (1.f - cur_prob);

        int block_width = l.dropblock_size_abs * multiplier;
        int block_height = l.dropblock_size_abs * multiplier;

        if (l.dropblock_size_rel) {
            block_width = l.dropblock_size_rel * l.w * multiplier;
            block_height = l.dropblock_size_rel * l.h * multiplier;
        }

        block_width = max_val_cmp(1, block_width);
        block_height = max_val_cmp(1, block_height);

        block_width = min_val_cmp(l.w, block_width);
        block_height = min_val_cmp(l.h, block_height);

        const int block_size = min_val_cmp(block_width, block_height);
        const float block_prob = cur_prob / (block_size*block_size);

        int num_blocks = l.batch * l.c;
        dropblock_fast_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (l.rand_gpu, block_prob, l.w, l.h, l.w*l.h, l.c, block_size, NULL, state.delta);
        CHECK_CUDA(cudaPeekAtLastError());

        num_blocks = get_number_of_blocks(l.outputs * l.batch, BLOCK);
        scale_dropblock_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (state.delta, l.outputs * l.batch, l.outputs, l.drop_blocks_scale_gpu);
        //scal_ongpu(l.outputs * l.batch, cur_scale, state.input, 1);
        //scal_ongpu(l.outputs * l.batch, l.drop_blocks_scale[0], state.input, 1);
        CHECK_CUDA(cudaPeekAtLastError());

        //drop_block_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (state.delta, size, l.rand_gpu, l.scale);
        //CHECK_CUDA(cudaPeekAtLastError());
    }
    // dropout
    else {
        yoloswag420blazeit360noscope << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (state.delta, size, l.rand_gpu, l.probability, l.scale);
        CHECK_CUDA(cudaPeekAtLastError());
    }
}
