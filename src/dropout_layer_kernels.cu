#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cstring>

#include "dropout_layer.h"
#include "dark_cuda.h"
#include "utils.h"

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

__global__ void drop_block_kernel(float *input, int size, float *mask, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id < size) input[id] = (mask[id]) ? 0 : (input[id] * scale);
}

void forward_dropout_layer_gpu(dropout_layer l, network_state state)
{
    if (!state.train) return;
    int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    //if (iteration_num < state.net.burn_in) return;

    // We gradually increase the block size and the probability of dropout - during the first half of the training
    float multiplier = 1.0;
    if(iteration_num < (state.net.max_batches / 2))
        multiplier = (iteration_num / (float)(state.net.max_batches / 2));

    // dropblock
    if (l.dropblock) {
        //l.probability = 1 / keep_prob
        const int max_blocks_per_channel = 10;
        const float cur_prob = l.probability * multiplier;

        int block_width = l.dropblock_size_abs * multiplier;
        int block_height = l.dropblock_size_abs * multiplier;

        if (l.dropblock_size_rel) {
            block_width = l.dropblock_size_rel * l.w * multiplier;
            block_height = l.dropblock_size_rel * l.h * multiplier;
        }

        block_width = max_val_cmp(1, block_width);
        block_height = max_val_cmp(1, block_height);

        const float part_occupied_by_block = block_width * block_height / ((float)l.w * l.h);
        const float prob_place_block = cur_prob / (part_occupied_by_block * max_blocks_per_channel);

        memset(l.rand, 0, l.batch * l.outputs * sizeof(float));

        float count_ones = 0;

        int b, k, x, y, i;
        for (b = 0; b < l.batch; b++) {
            for (k = 0; k < l.c; k++) {
                for (i = 0; i < max_blocks_per_channel; i++) {
                    float rnd = random_float_fast();
                    //printf(" rnd = %f \n", rnd);
                    //int rn = rand_int_fast(1, 7);
                    //printf(" rnd = %d \n", rn);
                    if (rnd < prob_place_block) {
                        //count_ones += block_width  *block_height;
                        const int pre_index = k*l.w*l.h + b*l.w*l.h*l.c;
                        const int x_block = rand_int_fast(0, l.w - block_width - 1);
                        const int y_block = rand_int_fast(0, l.h - block_height - 1);
                        for (y = y_block; y < (y_block + block_height); y++) {
                            memset(&l.rand[x_block + y*l.w + pre_index], 1, block_width * sizeof(float));
                            //for (x = x_block; x < (x_block + block_width); x++) {
                            //    const int index = x + y*l.w + pre_index;
                            //    l.rand[index] = 1;
                            //}
                        }
                    }
                }
            }
        }

        for (i = 0; i < (l.batch*l.outputs); ++i) if (l.rand[i]) count_ones++;

        cuda_push_array(l.rand_gpu, l.rand, l.batch*l.outputs);

        l.scale = (float)(l.batch*l.outputs) / (l.batch*l.outputs - count_ones);


        //printf("\n l.scale = %f, cur_prob = %f, count_ones = %f, prob_place_block = %f, block_width = %d, block_height = %d \n",
        //    l.scale, cur_prob, count_ones, prob_place_block, block_width, block_height);

        int size = l.inputs*l.batch;

        drop_block_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (state.input, size, l.rand_gpu, l.scale);
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
    //int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    //if (iteration_num < state.net.burn_in) return;

    int size = l.inputs*l.batch;

    // dropblock
    if (l.dropblock) {
        drop_block_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (state.delta, size, l.rand_gpu, l.scale);
        CHECK_CUDA(cudaPeekAtLastError());
    }
    // dropout
    else {
        yoloswag420blazeit360noscope << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (state.delta, size, l.rand_gpu, l.probability, l.scale);
        CHECK_CUDA(cudaPeekAtLastError());
    }
}
