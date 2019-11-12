#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "dark_cuda.h"
#include "box.h"


__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(x, n, binary);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>>(input, n, size, binary);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for (i = 0; i < size; ++i) {
        mean += fabs(weights[f*size + i]);
    }
    mean = mean / size;
    for (i = 0; i < size; ++i) {
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(weights, n, size, binary);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void set_zero_kernel(float *src, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) src[i] = 0;
}

__inline__ __device__
float warpAllReduceSum(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
#if CUDART_VERSION >= 9000
        val += __shfl_xor_sync(0xffffffff, val, mask);
#else
        val += __shfl_xor(val, mask);
#endif
    return val;
}

// only if (size % 32 == 0)
__global__ void reduce_kernel(float *weights, int n, int size, float *mean_arr_gpu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int f = i / size;
    if (f >= n) return;
    float warp_mean = warpAllReduceSum(fabs(weights[i]));
    if(i % 32 == 0)
        atomicAdd(&mean_arr_gpu[f], warp_mean / size);
}

__global__ void binarize_weights_mean_kernel(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int f = i / size;
    if (f >= n) return;
    float mean = mean_arr_gpu[f];
    binary[i] = (weights[i] > 0) ? mean : -mean;
}

void fast_binarize_weights_gpu(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
    if (size % 32 == 0) {
        size_t gridsize = n * size;
        const int num_blocks = get_number_of_blocks(gridsize, BLOCK);// gridsize / BLOCK + 1;

        set_zero_kernel << <(n/BLOCK + 1), BLOCK, 0, get_cuda_stream() >> > (mean_arr_gpu, n);
        reduce_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (weights, n, size, mean_arr_gpu);
        binarize_weights_mean_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (weights, n, size, binary, mean_arr_gpu);
        CHECK_CUDA(cudaPeekAtLastError());
    }
    else {
        binarize_weights_gpu(weights, n, size, binary);
    }
}


__global__ void cuda_f32_to_f16(float* input_f32, size_t size, half *output_f16)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_f16[idx] = __float2half(input_f32[idx]);
    //if (idx < size) output_f16[idx] = __float2half_rn(input_f32[idx]); // can't be compiled on Linux without casting
    // __float2half_ru, __float2half_rd, __float2half_rz, __float2half_rn
    //if (idx < size) *((unsigned short *)output_f16 + idx) = __float2half(input_f32[idx]);
}

void cuda_convert_f32_to_f16(float* input_f32, size_t size, float *output_f16) {
    cuda_f32_to_f16 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (half *)output_f16);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void cuda_f16_to_f32(half* input_f16, size_t size, float *output_f32)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_f32[idx] = __half2float(input_f16[idx]);
    //if (idx < size) output_f32[idx] = __half2float(*((unsigned short *)input_f16 + idx));
}

void cuda_convert_f16_to_f32(float* input_f16, size_t size, float *output_f32) {
    cuda_f16_to_f32 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> ((half *)input_f16, size, output_f32);
    CHECK_CUDA(cudaPeekAtLastError());
}

half *cuda_make_f16_from_f32_array(float *src, size_t n)
{
    half *dst16;
    size_t size = sizeof(half)*n;
    CHECK_CUDA(cudaMalloc((void **)&dst16, size));
    if (src) {
        assert(n > 0);
        cuda_convert_f32_to_f16(src, n, (float *)dst16);
    }
    if (!dst16) error("Cuda malloc failed\n");
    return dst16;
}

void forward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    //fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, (l.c / l.groups)*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        if (!l.align_bit_weights_gpu || state.train) {
            //binarize_weights_gpu(l.weights_gpu, l.n, (l.c / l.groups)*l.size*l.size, l.binary_weights_gpu);

            fast_binarize_weights_gpu(l.weights_gpu, l.n, (l.c / l.groups)*l.size*l.size, l.binary_weights_gpu, l.mean_arr_gpu);
        }

        if (l.align_bit_weights_gpu && !state.train && l.c >= 32 && l.stride_x == l.stride_y)
        {
            //return;
            //cudaError_t status = cudaSuccess;
            //int input_size = l.c*l.h*l.w*l.batch;

            int m = l.n / l.groups;
            int k = l.size*l.size*l.c / l.groups;
            int n = l.out_w*l.out_h;
            //float * a = l.weights_gpu;

            // int i, j;
            // for(i = 0; i < l.batch; ++i){
            // for (j = 0; j < l.groups; ++j) {

            int ldb_align = l.lda_align;
            size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
            //size_t t_intput_size = new_ldb * n;
            //size_t t_bit_input_size = t_intput_size / 8;// +1;

            if (l.c % 32 == 0)
            {
                //printf("\n\n l.index = %d, l.w = %d, l.c = %d, l.n = %d, l.stride = %d, l.pad = %d - new XNOR \n", l.index, l.w, l.c, l.n, l.stride, l.pad);
                //printf("l.align_workspace_size = %d, (l.c * l.w * l.h)  = %d \n", l.align_workspace_size, (l.c * l.w * l.h));

                //float *intput_cpu = (float *)calloc(l.inputs, sizeof(float));
                // state.input
                //cudaMemcpy(intput_cpu, state.input, l.inputs * sizeof(float), cudaMemcpyDefault);

                int ldb_align = l.lda_align;
                size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
                //size_t t_intput_size = new_ldb * l.bit_align;// n;
                //size_t t_bit_input_size = t_intput_size / 8;// +1;

                const int new_c = l.c / 32;

                //float *re_packed_input = (float *)calloc(l.c * l.w * l.h, sizeof(float));
                //uint32_t *bin_re_packed_input = (uint32_t *)calloc(new_c * l.w * l.h + 1, sizeof(uint32_t));

                // float32x4 by channel (as in cuDNN)
                //repack_input(intput_cpu, re_packed_input, l.w, l.h, l.c);


                // 32 x floats -> 1 x uint32_t
                //float_to_bit(re_packed_input, (uint8_t *)bin_re_packed_input, l.c * l.w * l.h);

                //cudaDeviceSynchronize();
                //start_timer();

                repack_input_gpu_bin(state.input, (uint32_t *)l.align_workspace_gpu, l.w, l.h, l.c);

                //repack_input_gpu(state.input, state.workspace, l.w, l.h, l.c);

                // 32 x floats -> 1 x uint32_t
                //float_to_bit_gpu(state.workspace, (unsigned char *)l.align_workspace_gpu, l.c * l.w * l.h);// l.align_workspace_size);

                //cudaDeviceSynchronize();
                //stop_timer_and_show_name("repack_input_gpu + float_to_bit_gpu");

                //free(re_packed_input);

                // slow - convolution the packed inputs and weights: float x 32 by channel (as in cuDNN)
                //convolution_repacked((uint32_t *)bin_re_packed_input, (uint32_t *)l.align_bit_weights, l.output,
                //    l.w, l.h, l.c, l.n, l.size, l.pad, l.new_lda, l.mean_arr);

                // // then exit from if()

                //float *b = state.workspace;
                //float *b = (float *)calloc(100 * 1024 * 1024, sizeof(float));
                //float *c = l.output;
                //memset(c, 0, l.outputs * sizeof(float));


                //im2col_cpu_custom((float *)bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, b);

                //cudaMemcpy(l.align_workspace_gpu, bin_re_packed_input, (new_c * l.w * l.h + 1) * sizeof(uint32_t), cudaMemcpyDefault);

                //start_timer();
                im2col_ongpu(l.align_workspace_gpu, new_c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
                //cudaDeviceSynchronize();
                //stop_timer_and_show_name("im2col_ongpu");

                //free(bin_re_packed_input);

                int new_k = l.size*l.size*l.c / 32;

                // good for (l.c == 64)
                //gemm_nn_bin_32bit_packed(m, n, new_k, 1,
                //    l.align_bit_weights, l.new_lda/32,
                //    b, n,
                //    c, n, l.mean_arr);

                // // then exit from if()


                //size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
                //size_t t_intput_size = new_ldb * l.bit_align;// n;
                //size_t t_bit_input_size = t_intput_size / 8;// +1;

                //char *t_bit_input = (char *)calloc(t_bit_input_size, sizeof(char));
                //transpose_uint32((uint32_t *)b, (uint32_t *)t_bit_input, new_k, n, n, new_ldb);
                //cudaMemcpy(l.transposed_align_workspace_gpu, t_bit_input, t_bit_input_size * sizeof(char), cudaMemcpyDefault);

                //cudaMemcpy(state.workspace, b, t_bit_input_size * sizeof(char), cudaMemcpyDefault);
                //printf("\n n = %d, n % 32 = %d, new_ldb = %d, new_ldb % 32 = %d \n", n, n % 32, new_ldb, new_ldb % 32);

                //start_timer();
                transpose_uint32_gpu((uint32_t *)state.workspace, (uint32_t *)l.transposed_align_workspace_gpu, new_k, n, n, new_ldb);
                //cudaDeviceSynchronize();
                //stop_timer_and_show_name("transpose_uint32_gpu");

                //cudaDeviceSynchronize();
                //stop_timer_and_show_name("repack_input_gpu_bin + im2col_ongpu + transpose_uint32_gpu_2");

                //start_timer();
                gemm_nn_custom_bin_mean_transposed_gpu(m, n, k,
                    (unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu,
                    new_ldb, l.output_gpu, n, l.mean_arr_gpu, l.biases_gpu, l.activation == LEAKY,
                    l.bin_conv_shortcut_in_gpu, l.bin_conv_shortcut_out_gpu);
                //cudaDeviceSynchronize();
                //stop_timer_and_show_name("gemm_nn_custom_bin_mean_transposed_gpu");


                // the main GEMM function
                //gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (uint8_t *)l.align_bit_weights, new_ldb, (uint8_t *)t_bit_input, new_ldb, c, n, l.mean_arr);

                //add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

                //cudaMemcpy(l.output_gpu, l.output, l.outputs * sizeof(float), cudaMemcpyDefault);


                // // alternative GEMM
                //gemm_nn_bin_transposed_32bit_packed(m, n, new_k, 1,
                //    l.align_bit_weights, l.new_lda/32,
                //    t_bit_input, new_ldb / 32,
                //    c, n, l.mean_arr);

                //free(t_bit_input);

                //free(b);
            }
            else
            {
                //printf("\n\n l.index = %d, l.w = %d, l.c = %d, l.n = %d, l.stride = %d, l.pad = %d - old XNOR \n", l.index, l.w, l.c, l.n, l.stride, l.pad);
                //cudaDeviceSynchronize();

                int i = 0;
                /*
                // if (l.stride == 1 && l.c >= 256 && l.size > 1)
                if (l.stride == 1 && l.c >= 1024 && l.size > 1 && 0)// && l.w >= 13) // disabled
                {
                    // stride=1 only
                    //start_timer();
                    im2col_align_bin_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, state.workspace, l.bit_align);
                    //cudaDeviceSynchronize();
                    //stop_timer_and_show_name("im2col_align_bin_ongpu");
                }
                else*/
                {
                    //start_timer();
                    im2col_align_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, l.align_workspace_gpu, l.bit_align);
                    //cudaDeviceSynchronize();
                    //stop_timer_and_show_name("im2col_align_ongpu");
                    //getchar();

                    // should be optimized
                    //start_timer();
                    float_to_bit_gpu(l.align_workspace_gpu, (unsigned char *)state.workspace, l.align_workspace_size);
                    //cudaDeviceSynchronize();
                    //stop_timer_and_show_name("float_to_bit_gpu");
                }
                //start_timer();
                transpose_bin_gpu((unsigned char *)state.workspace, (unsigned char *)l.transposed_align_workspace_gpu, k, n, l.bit_align, new_ldb, 8);
                //cudaDeviceSynchronize();
                //stop_timer_and_show_name("transpose_bin_gpu");

                //cudaDeviceSynchronize();
                //stop_timer_and_show_name("im2col_align_ongpu + float_to_bit_gpu + transpose_bin_gpu");

                // should be optimized
                //if(0) {//if (k > 1000) {    // sequentially input-shared - BAD
                //    gemm_nn_custom_bin_mean_transposed_sequentially_gpu(m, n, k,
                //        (unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu, new_ldb, l.output_gpu, n, l.mean_arr_gpu);
                //}
                //else {  // coalescing & weights-shared-memory - GOOD
                    //start_timer();
                    gemm_nn_custom_bin_mean_transposed_gpu(m, n, k,
                        (unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu,
                        new_ldb, l.output_gpu, n, l.mean_arr_gpu, l.biases_gpu, l.activation == LEAKY,
                        l.bin_conv_shortcut_in_gpu, l.bin_conv_shortcut_out_gpu);
                    //cudaDeviceSynchronize();
                    //stop_timer_and_show_name("gemm_nn_custom_bin_mean_transposed_gpu");
                //}
                //cudaDeviceSynchronize();
                //check_error(status);
                //getchar();
            }


            /*
            {
                float_to_bit_gpu(state.input, (unsigned char *)l.align_workspace_gpu, input_size);
                convolve_bin_gpu(l.align_workspace_gpu, (float *)l.align_bit_weights_gpu, l.output_gpu, l.w, l.h, l.c, l.n, l.size, l.pad, l.new_lda, l.mean_arr_gpu);

                //convolve_gpu(state.input, l.weights_gpu, l.output_gpu, l.w, l.h, l.c, l.n, l.size, l.pad);

                //cudaDeviceSynchronize();
                //check_error(status);

                add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
            }
            */

            //add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
            if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
            else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
            else if (l.activation != LINEAR && l.activation != LEAKY) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
            //if(l.activation != LINEAR && l.activation != LEAKY) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
            //if (l.binary || l.xnor) swap_binary(&l);
            //cudaDeviceSynchronize();
            return;
        }
    }

    if (l.xnor) {
        swap_binary(&l);
        binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        state.input = l.binary_input_gpu;
    }

    //fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

#ifdef CUDNN
    //float one = 1;    // alpha[0], beta[0] is float for HALF and FLOAT
    float alpha = 1, beta = 0;

//#ifdef CUDNN_HALF
    //if (state.use_mixed_precision) {
    int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    if (state.index != 0 && state.net.cudnn_half && !l.xnor && (!state.train || iteration_num > 3*state.net.burn_in) &&
        (l.c / l.groups) % 8 == 0 && l.n % 8 == 0 && !state.train)
    {
        //printf("\n CUDNN_HALF!!! state.index = %d \n", state.index);

        // Note: For improved performance it is advised to use beta[0] = 0.0.
        // For Tensor Core: cudnnSetConvolutionMathType() where cudnnMathType_t mathType = CUDNN_TENSOR_OP_MATH;
        // 1. or CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM and use CUDNN_DATA_HALF
        // 2. or CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
        // More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops

        const size_t input16_size = l.batch*l.c*l.w*l.h;
        const size_t output16_size = l.batch*l.out_c*l.out_h*l.out_w;

        if (*state.net.max_input16_size < input16_size) {
            //printf("\n input16_size: cur = %zu \t max = %zu \n", input16_size, *state.net.max_input16_size);
            *state.net.max_input16_size = input16_size;
            if (*state.net.input16_gpu) cuda_free(*state.net.input16_gpu);
            assert(*state.net.max_input16_size > 0);
            *state.net.input16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_input16_size);
        }
        float *input16 = *state.net.input16_gpu;

        if (*state.net.max_output16_size < output16_size) {
            *state.net.max_output16_size = output16_size;
            if (*state.net.output16_gpu) cuda_free(*state.net.output16_gpu);
            assert(*state.net.max_output16_size > 0);
            *state.net.output16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_output16_size);
        }
        float *output16 = *state.net.output16_gpu;

        assert(input16_size > 0);
        cuda_convert_f32_to_f16(state.input, input16_size, input16);

        //fill_ongpu(output16_size / 2, 0, (float *)output16, 1);
        CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
            &alpha,
            l.srcTensorDesc16,
            input16,
            l.weightDesc16,
            l.weights_gpu16,
            l.convDesc,
            l.fw_algo16,
            state.workspace,
            l.workspace_size,
            &beta,
            l.dstTensorDesc16,
            output16));


        if (l.batch_normalize)
        {
            if (state.train) // Training
            {
                simple_copy_ongpu(l.outputs*l.batch / 2, output16, l.x_gpu);
                //copy_ongpu(l.outputs*l.batch / 2, output16, 1, l.x_gpu, 1);
                //cudaMemcpyAsync(l.x_gpu, output16, l.outputs*l.batch*sizeof(half), cudaMemcpyDefault, get_cuda_stream());
                float one = 1.0f;
                float zero = 0.0f;
                // Batch-normalization can still take FP16 inputs and outputs, saving half the bandwidth
                // compared to FP32, it's just that the statistics and value adjustment should be done in FP32.
                CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                    CUDNN_BATCHNORM_SPATIAL,
                    &one,
                    &zero,
                    l.normDstTensorDescF16,
                    l.x_gpu,            // input
                    l.normDstTensorDescF16,
                    output16,            // output
                    l.normTensorDesc,
                    l.scales_gpu,       // input
                    l.biases_gpu,       // input
                    .01,
                    l.rolling_mean_gpu,        // input/output (should be FP32)
                    l.rolling_variance_gpu,    // input/output (should be FP32)
                    .00001,
                    l.mean_gpu,            // output (should be FP32) - optional cache to speedup cudnnBatchNormalizationBackward()
                    l.variance_gpu));    // output (should be FP32) - optional cache to speedup cudnnBatchNormalizationBackward()

                cuda_convert_f16_to_f32(output16, output16_size, l.output_gpu);
                //forward_batchnorm_layer_gpu(l, state);
            }
            else // Detection
            {
                cuda_convert_f16_to_f32(output16, output16_size, l.output_gpu);
                normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
                scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
                add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
            }
        }
        else // BIAS only
        {
            cuda_convert_f16_to_f32(output16, output16_size, l.output_gpu);
            add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
        }
    }
    else {

        //#else
        /*
        int input_nan_inf = is_nan_or_inf(state.input, l.inputs * l.batch);
        printf("\n is_nan_or_inf(state.input) = %d \n", input_nan_inf);
        if (input_nan_inf) getchar();

        int weights_nan_inf = is_nan_or_inf(l.weights_gpu, l.nweights);
        printf("\n is_nan_or_inf(l.weights_gpu) = %d \n", weights_nan_inf);
        if (weights_nan_inf) getchar();
        */

        CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
            &alpha, //&one,
            l.srcTensorDesc,
            state.input,
            l.weightDesc,
            l.weights_gpu,
            l.convDesc,
            l.fw_algo,
            state.workspace,
            l.workspace_size,
            &beta,  //&one,
            l.dstTensorDesc,
            l.output_gpu));

        //cudaDeviceSynchronize();
        if (l.batch_normalize) {
            forward_batchnorm_layer_gpu(l, state);
        }
        else {
            add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
        }
    //#endif    // CUDNN_HALF
    }


#else
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int i, j;
    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for (j = 0; j < l.groups; ++j) {
            //float *im = state.input + i*l.c*l.h*l.w;
            float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;
            float *a = l.weights_gpu + j*l.nweights / l.groups;
            float *b = state.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            if (l.size == 1) {
                b = im;
            }
            else {
                //im2col_ongpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, state.workspace);

                im2col_gpu_ext(im,          // input
                    l.c / l.groups,         // input channels
                    l.h, l.w,               // input size (h, w)
                    l.size, l.size,         // kernel size (h, w)
                    l.pad, l.pad,           // padding (h, w)
                    l.stride_y, l.stride_x,     // stride (h, w)
                    l.dilation, l.dilation, // dilation (h, w)
                    state.workspace);       // output

            }
            //gemm_ongpu(0, 0, m, n, k, 1., a, k, b, n, 1., c + i*m*n, n);
            gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
    }

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    }
    else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
#endif

//#ifndef CUDNN_HALF
//#endif // no CUDNN_HALF

    if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation != LINEAR) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
    //cudaDeviceSynchronize();    // for correct profiling of performance

    if (state.net.try_fix_nan) {
        fix_nan_and_inf(l.output_gpu, l.outputs*l.batch);
    }

    if(l.assisted_excitation && state.train) assisted_excitation_forward_gpu(l, state);

    if (l.antialiasing) {
        network_state s = { 0 };
        s.train = state.train;
        s.workspace = state.workspace;
        s.net = state.net;
        if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
        s.input = l.output_gpu;
        forward_convolutional_layer_gpu(*(l.input_layer), s);
        simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.input_antialiasing_gpu);
        simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.input_layer->output_gpu, l.output_gpu);
    }
}

void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    if (l.antialiasing) {
        network_state s = { 0 };
        s.train = state.train;
        s.workspace = state.workspace;
        s.net = state.net;
        s.delta = l.delta_gpu;  // s.delta will be returned to l.delta_gpu
        s.input = l.input_antialiasing_gpu;
        //if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
        simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.delta_gpu, l.input_layer->delta_gpu);
        backward_convolutional_layer_gpu(*(l.input_layer), s);

        simple_copy_ongpu(l.outputs*l.batch, l.input_antialiasing_gpu, l.output_gpu);
    }

    if(state.net.try_fix_nan) constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);

    if (l.activation == SWISH) gradient_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
    else if (l.activation == MISH) gradient_array_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
    else gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if (!l.batch_normalize)
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

//#ifndef CUDNN_HALF
    //if(l.batch_normalize){
    //    backward_batchnorm_layer_gpu(l, state);
    //} else {
    //    //backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    //}
//#endif // no CUDNN_HALF
    float *original_input = state.input;

    if(l.xnor) state.input = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1.f;
    float alpha = 1, beta = 0;

//#ifdef CUDNN_HALF
    int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    if (state.index != 0 && state.net.cudnn_half && !l.xnor && (!state.train || iteration_num > 3*state.net.burn_in) &&
        (l.c / l.groups) % 8 == 0 && l.n % 8 == 0 && !state.train)
    {
        const size_t input16_size = l.batch*l.c*l.w*l.h;
        const size_t delta16_size = l.batch*l.n*l.out_w*l.out_h;

        if (*state.net.max_input16_size < input16_size) {
            *state.net.max_input16_size = input16_size;
            if (*state.net.input16_gpu) cuda_free(*state.net.input16_gpu);
            assert(*state.net.max_input16_size > 0);
            *state.net.input16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_input16_size);
        }
        float *input16 = *state.net.input16_gpu;

        if (*state.net.max_output16_size < delta16_size) {
            *state.net.max_output16_size = delta16_size;
            if (*state.net.output16_gpu) cuda_free(*state.net.output16_gpu);
            assert(*state.net.max_output16_size > 0);
            *state.net.output16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_output16_size);
        }
        float *delta16 = *state.net.output16_gpu;

        assert(input16_size > 0);
        assert(delta16_size > 0);
        cuda_convert_f32_to_f16(state.input, input16_size, input16);
        cuda_convert_f32_to_f16(l.delta_gpu, delta16_size, delta16);

        if (l.batch_normalize) {
            //if (!state.train) {
            //    l.mean_gpu = l.rolling_mean_gpu;
            //    l.variance_gpu = l.rolling_variance_gpu;
            //}
            float one = 1.0f;
            float zero = 0.0f;
            CHECK_CUDNN(cudnnBatchNormalizationBackward(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                &one,
                &one,
                l.normDstTensorDescF16,
                l.x_gpu,                // input (input in BN-forward-inference)
                l.normDstTensorDescF16,
                delta16,                // input
                l.normDstTensorDescF16,
                l.x_norm_gpu,            // output (new delta)
                l.normTensorDesc,
                l.scales_gpu,            // input (should be FP32)
                l.scale_updates_gpu,    // output (should be FP32)
                l.bias_updates_gpu,        // output (should be FP32)
                .00001,
                l.mean_gpu,                // input (should be FP32)
                l.variance_gpu));        // input (should be FP32)

            simple_copy_ongpu(l.outputs*l.batch / 2, l.x_norm_gpu, delta16);
            //copy_ongpu(l.outputs*l.batch / 2, l.x_norm_gpu, 1, delta16, 1);
            //cudaMemcpyAsync(delta16, l.x_norm_gpu, l.outputs*l.batch * sizeof(half), cudaMemcpyDefault, get_cuda_stream());
        }
        else
        {
            //backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
        }

        // convert input: state.input (x), l.delta_gpu (y) from fp32 to fp16
        // get output: l.weight_updates_gpu (dw) and convert it to fp32 (ONLY if it is fp16)

        // calculate conv weight updates
        // Already: l.weight_updates_gpu = (l.weight_updates_gpu - l.weight*decay*batch*subdivision)*momentum
        //   so we should copy f32 to f16, or compute: f16=(w_up - w*d*b*s)*m
        assert((l.nweights) > 0);
        cuda_convert_f32_to_f16(l.weight_updates_gpu, l.nweights, l.weight_updates_gpu16);

        CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc16,
            input16, //state.input,
            l.ddstTensorDesc16,
            delta16, //l.delta_gpu,
            l.convDesc,
            l.bf_algo16,
            state.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc16,
            l.weight_updates_gpu16));    // l.weight_updates_gpu);

        cuda_convert_f16_to_f32(l.weight_updates_gpu16, l.nweights, l.weight_updates_gpu);

        if (state.delta) {
            if (l.binary || l.xnor) swap_binary(&l);

            // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
            // calculate delta for the next layer
            // convert input: l.weights_gpu (w), l.delta_gpu (dy) from fp32 to fp16
            // get output: state.delta (dx) and convert it to fp32 (ONLY if it is fp16)
            CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
                &alpha,
                l.weightDesc16,
                l.weights_gpu16, //l.weights_gpu,
                l.ddstTensorDesc16,
                delta16, //l.delta_gpu,
                l.convDesc,
                l.bd_algo16,
                state.workspace,
                l.workspace_size,
                &beta,
                l.dsrcTensorDesc16,
                input16));    // state.delta);

            cuda_convert_f16_to_f32(input16, input16_size, state.delta);

            if (l.binary || l.xnor) swap_binary(&l);
            if (l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
        }
    }
    else {
        //#else    // CUDNN_HALF

        if(l.batch_normalize){
            backward_batchnorm_layer_gpu(l, state);
        }

        // calculate conv weight updates
        // if used: beta=1 then loss decreases faster
        CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            state.input,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu));

        if (state.delta) {
            if (l.binary || l.xnor) swap_binary(&l);
            // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
            // calculate delta for the next layer
            CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                state.delta));

            if (l.binary || l.xnor) swap_binary(&l);
            if (l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
        }
    }

//#endif    // CUDNN_HALF

#else    // CUDNN
    if (l.batch_normalize) {
        backward_batchnorm_layer_gpu(l, state);
    }

    int m = l.n / l.groups;
    int n = l.size*l.size*l.c / l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for (j = 0; j < l.groups; ++j) {
            float * a = l.delta_gpu + (i*l.groups + j)*m*k;
            float * b = state.workspace;
            float * c = l.weight_updates_gpu + j*l.nweights / l.groups;

            float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

            //im2col_ongpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
            im2col_gpu_ext(im,          // input
                l.c / l.groups,         // input channels
                l.h, l.w,               // input size (h, w)
                l.size, l.size,         // kernel size (h, w)
                l.pad, l.pad,           // padding (h, w)
                l.stride_y, l.stride_x,     // stride (h, w)
                l.dilation, l.dilation, // dilation (h, w)
                state.workspace);       // output
            //gemm_ongpu(0, 1, m, n, k, 1, a + i*m*k, k, b, k, 1, c, n);
            gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

            if (state.delta) {
                if (l.binary || l.xnor) swap_binary(&l);
                float * a = l.weights_gpu + j*l.nweights / l.groups;
                float * b = l.delta_gpu + (i*l.groups + j)*m*k;
                float * c = state.workspace;

                //gemm_ongpu(1, 0, n, k, m, 1, a, n, b + i*k*m, k, 0, c, k);
                gemm_ongpu(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);


                float *delta = state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

                //col2im_ongpu(state.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, delta);
                col2im_gpu_ext(
                    state.workspace,        // input
                    l.c / l.groups,         // input channels
                    l.h, l.w,               // input size (h, w)
                    l.size, l.size,         // kernel size (h, w)
                    l.pad, l.pad,           // padding size (h, w)
                    l.stride_y, l.stride_x,     // stride size (h, w)
                    l.dilation, l.dilation, // dilation size (h, w)
                    delta);                 // output (delta)

                if (l.binary || l.xnor) {
                    swap_binary(&l);
                }
                if (l.xnor) gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
            }
        }
    }
#endif
    if (state.net.try_fix_nan) {
        if (state.delta) {
            fix_nan_and_inf(state.delta, l.inputs * l.batch);
        }
        int size = l.nweights;
        fix_nan_and_inf(l.weight_updates_gpu, size);
        fix_nan_and_inf(l.weights_gpu, size);
    }
}

__global__ void calc_avg_activation_kernel(float *src, float *dst, int size, int channels, int batches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int xy = i % size;
    int b = i / size;

    if (i < size*batches) {
        dst[i] = 0;
        for (int c = 0; c < channels; ++c) {
            dst[i] += src[xy + size*(c + channels*b)];
        }
        dst[i] = dst[i] / channels;
    }
}

void calc_avg_activation_gpu(float *src, float *dst, int size, int channels, int batches)
{
    const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

    calc_avg_activation_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (src, dst, size, channels, batches);
}


__global__ void assisted_activation_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int xy = i % size;
    int b = i / size;

    if (b < batches) {
        for (int c = 0; c < channels; ++c) {
            output[xy + size*(c + channels*b)] += alpha * gt_gpu[i] * a_avg_gpu[i];
            //output[xy + size*(c + channels*b)] += gt_gpu[i] * a_avg_gpu[i];
            //output[xy + size*(c + channels*b)] += gt_gpu[i] * output[xy + size*(c + channels*b)];
            //output[xy + size*(c + channels*b)] = a_avg_gpu[i];
        }
    }
}

void assisted_activation_gpu(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
    const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

    assisted_activation_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (alpha, output, gt_gpu, a_avg_gpu, size, channels, batches);
}


__global__ void assisted_activation2_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int xy = i % size;
    int b = i / size;
    float beta = 1 - alpha;

    if (b < batches) {
        for (int c = 0; c < channels; ++c) {
            if(gt_gpu[i] == 0)
                output[xy + size*(c + channels*b)] *= beta;

        }
    }
}

void assisted_activation2_gpu(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
    const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

    assisted_activation2_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (alpha, output, gt_gpu, a_avg_gpu, size, channels, batches);
}

void assisted_excitation_forward_gpu(convolutional_layer l, network_state state)
{
    const int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);

    // epoch
    //const float epoch = (float)(*state.net.seen) / state.net.train_images_num;

    // calculate alpha
    //const float alpha = (1 + cos(3.141592 * iteration_num)) / (2 * state.net.max_batches);
    //const float alpha = (1 + cos(3.141592 * epoch)) / (2 * state.net.max_batches);
    float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches)) / 2;
    //float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches));

    if (l.assisted_excitation == 1) {
        if (iteration_num > state.net.max_batches / 2) return;
    }
    else {
        if (iteration_num < state.net.burn_in) return;
        else
            if (iteration_num > l.assisted_excitation) return;
        else
            alpha = (1 + cos(3.141592 * iteration_num / (state.net.burn_in + l.assisted_excitation))) / 2; // from 1 to 0
    }

    //printf("\n epoch = %f, alpha = %f, seen = %d, max_batches = %d, train_images_num = %d \n",
    //    epoch, alpha, (*state.net.seen), state.net.max_batches, state.net.train_images_num);

    //const int size = l.outputs * l.batch;

    float *a_avg = (float *)calloc(l.out_w * l.out_h * l.batch, sizeof(float));
    float *gt = (float *)calloc(l.out_w * l.out_h * l.batch, sizeof(float));

    int b;
    int w, h;

    l.max_boxes = state.net.num_boxes;
    l.truths = l.max_boxes*(4 + 1);

    int num_truth = l.batch*l.truths;
    float *truth_cpu = (float *)calloc(num_truth, sizeof(float));
    cuda_pull_array(state.truth, truth_cpu, num_truth);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());

    for (b = 0; b < l.batch; ++b)
    {
        // calculate G
        int t;
        for (t = 0; t < state.net.num_boxes; ++t) {
            box truth = float_to_box_stride(truth_cpu + t*(4 + 1) + b*l.truths, 1);
            if (!truth.x) break;  // continue;
            float beta = 0;
            //float beta = 1 - alpha; // from 0 to 1
            float dw = (1 - truth.w) * beta;
            float dh = (1 - truth.h) * beta;
            //printf(" alpha = %f, beta = %f, truth.w = %f, dw = %f, tw+dw = %f, l.out_w = %d \n", alpha, beta, truth.w, dw, truth.w+dw, l.out_w);

            int left = floor((truth.x - (dw + truth.w) / 2) * l.out_w);
            int right = ceil((truth.x + (dw + truth.w) / 2) * l.out_w);
            int top = floor((truth.y - (dh + truth.h) / 2) * l.out_h);
            int bottom = ceil((truth.y + (dh + truth.h) / 2) * l.out_h);
            if (left < 0) left = 0;
            if (top < 0) top = 0;
            if (right > l.out_w) right = l.out_w;
            if (bottom > l.out_h) bottom = l.out_h;

            for (w = left; w <= right; w++) {
                for (h = top; h < bottom; h++) {
                    gt[w + l.out_w * h + l.out_w*l.out_h*b] = 1;
                }
            }
        }
    }

    cuda_push_array(l.gt_gpu, gt, l.out_w * l.out_h * l.batch);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());

    // calc avg_output on GPU - for whole batch
    calc_avg_activation_gpu(l.output_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());

    // calc new output
    //assisted_activation2_gpu(1, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);  // AE3: gt increases (beta = 1 - alpha = 0)
    //assisted_activation2_gpu(alpha, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);
    assisted_activation_gpu(alpha, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());



    /*
    for (b = 0; b < l.batch; ++b)
    {
        // calculate average A
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++) {
                    a_avg[w + l.out_w*(h + l.out_h*b)] += l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))];
                }
                a_avg[w + l.out_w*(h + l.out_h*b)] /= l.out_c;  // a_avg / d
            }
        }
    }

    // change activation
    for (b = 0; b < l.batch; ++b)
    {
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++)
                {
                    // a = a + alpha(t) + e(c,i,j) = a + alpha(t) + g(i,j) * avg_a(i,j) / channels
                    l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] +=
                        alpha *
                        g[w + l.out_w*(h + l.out_h*b)] *
                        a_avg[w + l.out_w*(h + l.out_h*b)];

                    //l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] =
                    //    alpha * g[w + l.out_w*(h + l.out_h*b)] * a_avg[w + l.out_w*(h + l.out_h*b)];
                }
            }
        }
    }
    */

    if (0)   // visualize ground truth
    {
#ifdef OPENCV
        cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
        cudaStreamSynchronize(get_cuda_stream());
        CHECK_CUDA(cudaPeekAtLastError());

        for (b = 0; b < l.batch; ++b)
        {
            printf(" Assisted Excitation alpha = %f \n", alpha);
            image img = float_to_image(l.out_w, l.out_h, 1, &gt[l.out_w*l.out_h*b]);
            char buff[100];
            sprintf(buff, "a_excitation_gt_%d", b);
            show_image_cv(img, buff);

            //image img2 = float_to_image(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
            image img2 = float_to_image_scaled(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
            char buff2[100];
            sprintf(buff2, "a_excitation_output_%d", b);
            show_image_cv(img2, buff2);

            /*
            int c = l.out_c;
            if (c > 4) c = 4;
            image img3 = float_to_image(l.out_w, l.out_h, c, &l.output[l.out_w*l.out_h*l.out_c*b]);
            image dc = collapse_image_layers(img3, 1);
            char buff3[100];
            sprintf(buff3, "a_excitation_act_collapsed_%d", b);
            show_image_cv(dc, buff3);
            */

            wait_key_cv(5);
        }
        wait_until_press_key_cv();
#endif // OPENCV
    }

    free(truth_cpu);
    free(gt);
    free(a_avg);
}

void pull_convolutional_layer(convolutional_layer l)
{
    cuda_pull_array_async(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array_async(l.biases_gpu, l.biases, l.n);
    cuda_pull_array_async(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array_async(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array_async(l.scales_gpu, l.scales, l.n);
        cuda_pull_array_async(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array_async(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
    if (l.adam){
        cuda_pull_array_async(l.m_gpu, l.m, l.nweights);
        cuda_pull_array_async(l.v_gpu, l.v, l.nweights);
    }
    CHECK_CUDA(cudaPeekAtLastError());
    cudaStreamSynchronize(get_cuda_stream());
}

void push_convolutional_layer(convolutional_layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
#ifdef CUDNN_HALF
    assert(l.nweights > 0);
    cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
#endif
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    if (l.train) {
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    }
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
    if (l.adam){
        cuda_push_array(l.m_gpu, l.m, l.nweights);
        cuda_push_array(l.v_gpu, l.v, l.nweights);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}

void update_convolutional_layer_gpu(layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    float learning_rate = learning_rate_init*l.learning_rate_scale;
    //float momentum = a.momentum;
    //float decay = a.decay;
    //int batch = a.batch;

    fix_nan_and_inf(l.weight_updates_gpu, l.nweights);
    fix_nan_and_inf(l.weights_gpu, l.nweights);

    if (l.adam) {
        //adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.nweights, batch, l.t);

        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
        if (l.scales_gpu) {
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
        }
    }
    else {
        //axpy_ongpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        //axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        //scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);
        axpy_ongpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_ongpu(l.n, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);

        if (l.scales_gpu) {
            axpy_ongpu(l.n, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_ongpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    //if (l.clip) {
    //    constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    //}
}

/*
void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    if(layer.scales_gpu){
        axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
        scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);
    }

    if(layer.adam){
        scal_ongpu(size, layer.B1, layer.m_gpu, 1);
        scal_ongpu(size, layer.B2, layer.v_gpu, 1);

        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);

        axpy_ongpu(size, -(1-layer.B1), layer.weight_updates_gpu, 1, layer.m_gpu, 1);
        mul_ongpu(size, layer.weight_updates_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, (1-layer.B2), layer.weight_updates_gpu, 1, layer.v_gpu, 1);

        adam_gpu(size, layer.weights_gpu, layer.m_gpu, layer.v_gpu, layer.B1, layer.B2, learning_rate/batch, layer.eps, layer.t+1);
        fill_ongpu(size, 0, layer.weight_updates_gpu, 1);
    }else{
        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);  // wu = wu - w*decay*batch
        axpy_ongpu(size, learning_rate/batch, layer.weight_updates_gpu, 1, layer.weights_gpu, 1); // w = w + wu*lr/batch
        scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);    // wu = wu*momentum // wu = (wu - w*decay*batch)*momentum
        // w = w + (wu - w*decay*batch)*lr/batch = w + wu*lr/batch - w*decay*lr = w*(1-decay*lr) + wu*lr/batch
        //wu_prev = (wu_old - w_old*decay*batch)*momentum


        //weights_update = weights_update_new + (weights_update_old - weights_old*decay*batch)*momentum - weights_new*decay*batch =
        // = weights_update_new + weights_update_old*momentum - weights_old*decay*batch*momentum - weights_new*decay*batch
        // = weights_update_new + weights_update_old*momentum - (weights_old*momentum + weights_new)*decay*batch

        //------------- RESULT --------------
        // weights_update = weights_update_new + weights_update_old*momentum - (weights_old*momentum + weights_new)*decay*batch
        //-----------------------------------

        // weights_newest = weights_new + (weights_update_new + weights_update_old*momentum - (weights_old*momentum + weights_new)*decay*batch)*lr/batch
        // = weights_new + weights_update_new*lr/batch + weights_update_old*momentum*lr/batch - weights_old*momentum*decay*batch*lr/batch - weights_new*decay*batch*lr/batch
        // = weights_new + weights_update_new*lr/batch + weights_update_old*momentum*lr/batch - weights_old*momentum*decay*lr - weights_new*decay*lr
        // = weights_new*(1 - decay*lr) - weights_old*momentum*decay*lr + (weights_update_new + weights_update_old*momentum)*lr/batch

        //------------- RESULT --------------
        // weights_newest = weights_new*(1 - decay*lr) - weights_old*momentum*(decay*lr) + (weights_update_new + weights_update_old*momentum)*lr/batch =
        // = weights_new - (weights_new + weights_old*momentum)*decay*lr + (weights_update_new + weights_update_old*momentum)*lr / batch
        //-----------------------------------
    }
}
*/
