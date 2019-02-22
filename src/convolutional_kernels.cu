#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "cudnn.lib")
#endif
#endif

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"


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
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        if (!l.align_bit_weights_gpu || state.train) {
            //binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);

            fast_binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu, l.mean_arr_gpu);
        }
        //swap_binary(&l);
        //binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        //state.input = l.binary_input_gpu;
        //cudaDeviceSynchronize();

        if (l.align_bit_weights_gpu && !state.train && l.c >= 32)
        {
            //return;
            cudaError_t status = cudaSuccess;
            int input_size = l.c*l.h*l.w*l.batch;

            int m = l.n;
            int k = l.size*l.size*l.c;
            int n = l.out_w*l.out_h;
            float * a = l.weights_gpu;

            int ldb_align = l.lda_align;
            size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
            size_t t_intput_size = new_ldb * n;
            size_t t_bit_input_size = t_intput_size / 8;// +1;

            if (l.c % 32 == 0)
            {
                //printf("\n\n l.index = %d, l.w = %d, l.c = %d, l.n = %d, l.stride = %d, l.pad = %d - new XNOR \n", l.index, l.w, l.c, l.n, l.stride, l.pad);
                //printf("l.align_workspace_size = %d, (l.c * l.w * l.h)  = %d \n", l.align_workspace_size, (l.c * l.w * l.h));

                //float *intput_cpu = (float *)calloc(l.inputs, sizeof(float));
                // state.input
                //cudaMemcpy(intput_cpu, state.input, l.inputs * sizeof(float), cudaMemcpyDefault);

                int ldb_align = l.lda_align;
                size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
                size_t t_intput_size = new_ldb * l.bit_align;// n;
                size_t t_bit_input_size = t_intput_size / 8;// +1;

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
            if (l.activation != LINEAR && l.activation != LEAKY) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
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
    float one = 1;    // alpha[0], beta[0] is float for HALF and FLOAT
    float alpha = 1, beta = 0;

//#ifdef CUDNN_HALF
    //if (state.use_mixed_precision) {
    int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    if (state.index != 0 && state.net.cudnn_half && !l.xnor && (!state.train || iteration_num > 3*state.net.burn_in) &&
        l.c % 8 == 0 && l.n % 8 == 0)
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
                    l.scales_gpu,
                    l.biases_gpu,
                    .01,
                    l.rolling_mean_gpu,        // output (should be FP32)
                    l.rolling_variance_gpu,    // output (should be FP32)
                    .00001,
                    l.mean_gpu,            // output (should be FP32)
                    l.variance_gpu));    // output (should be FP32)

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

    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        float *im = state.input + i*l.c*l.h*l.w;
        float * a = l.weights_gpu;
        float * b = state.workspace;
        float * c = l.output_gpu;
        if (l.size == 1) {
            b = im;
        }
        else {
            im2col_ongpu(im, l.c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
        }
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
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

    if (l.activation != LINEAR) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
    //cudaDeviceSynchronize();    // for correct profiling of performance
}

void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

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
    float one = 1;
    float alpha = 1, beta = 0;

//#ifdef CUDNN_HALF
    int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    if (state.index != 0 && state.net.cudnn_half && !l.xnor && (!state.train || iteration_num > 3*state.net.burn_in) &&
        l.c % 8 == 0 && l.n % 8 == 0)
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
                l.x_gpu,                // input
                l.normDstTensorDescF16,
                delta16,                // input
                l.normDstTensorDescF16,
                l.x_norm_gpu,            // output
                l.normTensorDesc,
                l.scales_gpu,            // output (should be FP32)
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
        assert((l.c*l.n*l.size*l.size) > 0);
        cuda_convert_f32_to_f16(l.weight_updates_gpu, l.c*l.n*l.size*l.size, l.weight_updates_gpu16);

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

        cuda_convert_f16_to_f32(l.weight_updates_gpu16, l.c*l.n*l.size*l.size, l.weight_updates_gpu);

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

    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = state.workspace;
        float * c = l.weight_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            if(l.binary || l.xnor) swap_binary(&l);
            float * a = l.weights_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array_async(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array_async(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array_async(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array_async(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array_async(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array_async(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array_async(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_pull_array_async(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array_async(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
    CHECK_CUDA(cudaPeekAtLastError());
    cudaStreamSynchronize(get_cuda_stream());
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
#ifdef CUDNN_HALF
    assert((layer.c*layer.n*layer.size*layer.size) > 0);
    cuda_convert_f32_to_f16(layer.weights_gpu, layer.c*layer.n*layer.size*layer.size, layer.weights_gpu16);
#endif
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}

void update_convolutional_layer_gpu(layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    float learning_rate = learning_rate_init*l.learning_rate_scale;
    //float momentum = a.momentum;
    //float decay = a.decay;
    //int batch = a.batch;
    int size = l.size*l.size*l.c*l.n;   // old

    if (l.adam) {
        //adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, size, batch, l.t);

        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
        if (l.scales_gpu) {
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
        }
    }
    else {
        //axpy_ongpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        //axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        //scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);
        axpy_ongpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_ongpu(size, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_ongpu(size, momentum, l.weight_updates_gpu, 1);

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
