#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <stdint.h>

#include "im2col.h"
#include "cuda.h"

#include <stdio.h>
#include <assert.h>
//#include <cuda.h>


template<typename T1, typename T2>
__device__ inline T1 __shfl_custom(T1 val, T2 lane) {
#if CUDART_VERSION >= 9000
    return __shfl_sync(FULL_MASK, val, lane);
#else
    return __shfl(val, lane);
#endif
}

template<typename T>
__device__ inline uint32_t __ballot_custom(T val) {
#if CUDART_VERSION >= 9000
    return __ballot_sync(FULL_MASK, val);
#else
    return __ballot(val);
#endif
}


// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //data_im[(channel_in * height + h_in) * width + w_in + i * width + j];
                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK, 0, get_cuda_stream()>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);

    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------

/*
__global__ void im2col_align_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    float *data_col, const int bit_align)
{
    //__shared__ float tmp_s[1];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x*gridDim.x) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        //data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        data_col_ptr += channel_out * bit_align + h_out * width_col + w_out;
        float* data_col_ptr_32 = data_col + (channel_out * bit_align + h_out * width_col + w_out)/32;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                float val = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                *data_col_ptr = val;
                //tmp_s[0] = val;

                //*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                //    data_im_ptr[i * width + j] : 0;

                //float src_val = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
                //unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
                //if (threadIdx.x % WARP_SIZE == 0) *((unsigned int*)data_col_ptr_32) = bit_mask;
                // use atomicOr() // *dst_ptr |= (mask << (col_index % 8));
                //data_col_ptr_32 += bit_align / 32;

                //data_col_ptr += height_col * width_col;
                data_col_ptr += bit_align;
            }
        }
    }
}
*/

// float 32
__global__ void im2col_align_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    float *data_col, const int bit_align)
{
    //__shared__ float tmp_s[1];


    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x*gridDim.x) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        //float* data_col_ptr = data_col;
        //float* data_col_ptr_32 = data_col + (channel_out * bit_align + h_out * width_col + w_out) / 32;
        //data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        //data_col_ptr += channel_out * bit_align + h_out * width_col + w_out;
        float* data_col_ptr = &data_col[channel_out * bit_align + h_out * width_col + w_out];
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                float val = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                int pre_out_index = index % (width_col*height_col);
                int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;// h_out * width_col + w_out;
                data_col[out_index] = val;

                //*data_col_ptr = val;
                //dst_s[threadIdx.x] = val;
                //tmp_s[0] = val;

                //*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                //    data_im_ptr[i * width + j] : 0;

                //float src_val = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
                //unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
                //if (threadIdx.x % WARP_SIZE == 0) *((unsigned int*)data_col_ptr_32) = bit_mask;
                // use atomicOr() // *dst_ptr |= (mask << (col_index % 8));
                //data_col_ptr_32 += bit_align / 32;

                //data_col_ptr += height_col * width_col;
                data_col_ptr += bit_align;
            }
        }
    }
}

void im2col_align_ongpu(float *im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float *data_col, int bit_align) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_align_gpu_kernel << <(num_kernels + BLOCK - 1) / BLOCK,
        BLOCK, 0, get_cuda_stream() >> >(
            num_kernels, im, height, width, ksize, pad,
            stride, height_col,
            width_col, data_col, bit_align);

    CHECK_CUDA(cudaPeekAtLastError());
}


// --------------------------------



// binary im2col - stride=1
__global__ void im2col_align_bin_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize, const int channels,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    float *data_col, const int bit_align)
{
    __shared__ float tmp_s[1];
    __shared__ ulonglong4 tmp256_s[1];


    //#define SHRED_VALS ((BLOCK / 169) * )
    //__shared__ float dst_s[1024];
    //__shared__ float dst_s[1024];
    //__shared__ uint32_t bit_s[32];
    //__shared__ uint8_t bit_s[128];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    //for (; index < n; index += blockDim.x*gridDim.x)
    {
        int c_index = index;
        int channel_in = c_index % channels;

        //int h_out = index % height_col;
        //int c_index = index / height_col;
        //int channel_in = c_index % channels;

        int channel_out = channel_in * ksize * ksize;

        int j_index = c_index / channels;
        int j = j_index % ksize;
        int i = j_index / ksize;

        int pre_out_index = (channel_out + i*ksize + j) * bit_align;
        int j_pad = (j - pad);
        int i_pad = (i - pad);

        for(int wh_index = 0; wh_index < (height_col*width_col); wh_index += 32)
        //for (int h_out = 0; h_out < height_col; ++h_out)
        {

            // the end of padding
            //if(0)
            //for (int w_out = 0; w_out < (width_col); w_out += 32)
            {
                const int w_out = wh_index % width_col;
                const int h_out = wh_index / width_col;

                const int w = w_out + j_pad;
                const int h = h_out + i_pad;

                int pre_in_index = channel_in * height * width;
                int pre_in_wh_index = h * width + w;

                int send_wh_index = wh_index;
                if (i >= ksize) send_wh_index = height_col*width_col;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t)
                {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int cur_wh_index = __shfl_custom(send_wh_index, t) + lane_id;

                    if (cur_wh_index < (width_col*height_col))// && (cur_i_pad+pad) < ksize)
                    {
                        const int cur_pre_out_index = __shfl_custom(pre_out_index, t);

                        const int cur_pre_in_index = __shfl_custom(pre_in_index, t);
                        const int cur_pre_in_wh_index = __shfl_custom(pre_in_wh_index, t) + lane_id;

                        int w = cur_pre_in_wh_index % width;
                        int h = cur_pre_in_wh_index / width;
                        int in_index = cur_pre_in_index + cur_pre_in_wh_index;

                        int out_index = cur_pre_out_index + cur_wh_index;

                        float val = (w >= 0 && w < width && h >= 0 && h < height) ?
                            data_im[in_index] : float();

                        //data_col[out_index] = val;
                        //tmp_s[0] = val;

                        uint32_t bit_mask = __ballot_custom(val > 0);
                        if (lane_id == 0) {
                            uint8_t *bit8_ptr = &(((uint8_t *)data_col)[out_index / 8]);
                            uint32_t *bit32_ptr = (uint32_t *)bit8_ptr;
                            *bit32_ptr = bit_mask;
                        }
                    }


                }

            }// w_out

        }
    }
}


void im2col_align_bin_ongpu(float *im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float *data_col, int bit_align) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    //int num_kernels = channels * height_col * width_col * ksize * ksize;
    //int num_kernels = channels * ksize * ksize * height_col;
    int num_kernels = channels * ksize * ksize;
    int num_blocks = num_kernels / BLOCK + 1;

    //im2col_align_bin_gpu_kernel << <(num_kernels + BLOCK - 1) / BLOCK,
    im2col_align_bin_gpu_kernel << <num_blocks,
        BLOCK, 0, get_cuda_stream() >> >(
            num_kernels, im, height, width, ksize, channels, pad,
            stride, height_col,
            width_col, data_col, bit_align);

    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------

/*
__global__ void float_to_bit_gpu_kernel(float *src, unsigned char *dst, size_t size)
{
    //const int size_aligned = size + (WARP_SIZE - size % WARP_SIZE);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float src_val;

    //for (; index < size_aligned; index += blockDim.x*gridDim.x)
    {
        //src_val = src[index];
        if(index < size) src_val = src[index];
        else src_val = 0;
        //unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
        unsigned int bit_mask = __ballot_custom(src_val > 0);
        if (threadIdx.x % WARP_SIZE == 0) ((unsigned int*)dst)[index / 32] = bit_mask;
    }
}
*/

/*
__global__ void float_to_bit_gpu_kernel(float *src, unsigned char *dst, size_t size)
{
    //const int size_aligned = size + (WARP_SIZE - size % WARP_SIZE);
    __shared__ uint32_t tmp[WARP_SIZE];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float src_val;
    uint32_t *dst32_ptr = ((unsigned int*)dst);

    //for (; index < size_aligned; index += blockDim.x*gridDim.x)
    {
        //src_val = src[index];
        if (index < size) src_val = src[index];
        else src_val = 0;
        //unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
        const int num_of_warps = blockDim.x / WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        uint32_t bit_mask = __ballot_custom(src_val > 0);

        if (lane_id == 0) tmp[warp_id] = bit_mask;

        __syncthreads();
        if (warp_id == 0) {
            if (lane_id < num_of_warps) {
                dst32_ptr[index / 32 + lane_id] = tmp[lane_id];
            }
        }
        __syncthreads();
    }
}
*/

__global__ void float_to_bit_gpu_kernel(float *src, unsigned char *dst, size_t size)
{
    __shared__ uint32_t tmp[WARP_SIZE*32];

    int index = 32*blockIdx.x*blockDim.x + threadIdx.x;
    float src_val;
    uint32_t *dst32_ptr = ((unsigned int*)dst);

    int i;
    for(i = 0; i < 32; ++i)
    {
        if ((index + i * 1024) < size) src_val = src[index + i*1024];
        else src_val = 0;
        //unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
        const int num_of_warps = blockDim.x / WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        uint32_t bit_mask = __ballot_custom(src_val > 0);
        if (lane_id == 0) tmp[i * 32 + warp_id] = bit_mask;
    }
    __syncthreads();
    dst32_ptr[blockIdx.x*blockDim.x + threadIdx.x] = tmp[threadIdx.x];
}


void float_to_bit_gpu(float *src, unsigned char *dst, size_t size)
{
    //const int num_blocks = size / 1024 + 1;
    //const int num_blocks = size / (32*1024) + 1;
    const int num_blocks = get_number_of_blocks(size, 32 * 1024);
    float_to_bit_gpu_kernel<<<num_blocks, 1024, 0, get_cuda_stream()>>>(src, dst, size);
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------


__device__ __host__ static inline void remove_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] &= ~(1 << dst_shift);
}

__device__ __host__ static inline void set_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
    //dst[dst_i] |= 1 << (8 - dst_shift);
}

__device__ __host__ static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    //unsigned char val = (src[src_i] & (1 << (8 - src_shift))) > 0;
    return val;
}

// Intel CPUs and nVidia CUDA GPU are little endian
__device__ __host__ unsigned char reverse_byte(unsigned char a)
{
    return ((a & 0x1) << 7) | ((a & 0x2) << 5) |
        ((a & 0x4) << 3) | ((a & 0x8) << 1) |
        ((a & 0x10) >> 1) | ((a & 0x20) >> 3) |
        ((a & 0x40) >> 5) | ((a & 0x80) >> 7);
}

__device__ __host__ unsigned char reverse_byte_2(unsigned char a)
{
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

__device__ unsigned char reverse_byte_CUDA(unsigned char a)
{
    uint32_t tmp = __brev(a);
    return tmp >> 24;
}

__device__ void transpose8rS32_reversed_diagonale(unsigned char* A, unsigned char* B, int m, int n)
{
    unsigned x, y, t;

    // Load the array and pack it into x and y.
    x = (A[0] << 24) | (A[m] << 16) | (A[2 * m] << 8) | A[3 * m];
    y = (A[4 * m] << 24) | (A[5 * m] << 16) | (A[6 * m] << 8) | A[7 * m];

    t = (x ^ (x >> 7)) & 0x00AA00AA;  x = x ^ t ^ (t << 7);
    t = (y ^ (y >> 7)) & 0x00AA00AA;  y = y ^ t ^ (t << 7);

    t = (x ^ (x >> 14)) & 0x0000CCCC;  x = x ^ t ^ (t << 14);
    t = (y ^ (y >> 14)) & 0x0000CCCC;  y = y ^ t ^ (t << 14);

    t = (x & 0xF0F0F0F0) | ((y >> 4) & 0x0F0F0F0F);
    y = ((x << 4) & 0xF0F0F0F0) | (y & 0x0F0F0F0F);
    x = t;

    B[7 * n] = reverse_byte_CUDA(x >> 24);  B[6 * n] = reverse_byte_CUDA(x >> 16);  B[5 * n] = reverse_byte_CUDA(x >> 8);  B[4 * n] = reverse_byte_CUDA(x);
    B[3 * n] = reverse_byte_CUDA(y >> 24);  B[2 * n] = reverse_byte_CUDA(y >> 16);  B[1 * n] = reverse_byte_CUDA(y >> 8);  B[0 * n] = reverse_byte_CUDA(y);

    //__device__ ​ unsigned int 	__brev(unsigned int  x)
    //Reverse the bit order of a 32 bit unsigned integer.
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html
}


// transpose 8x8 bit
__global__ void transpose_bin_gpu_kernel(unsigned char *A, unsigned char *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    //for (i = 0; i < n; i += 8)
    {
        i = (index*8) % n;
        int j;
        //for (j = 0; j < m - 8; j += 8)
        {
            j = ((index * 8) / n) * 8;
            if (j < m) {
                int a_index = i*lda + j;
                int b_index = j*ldb + i;
                transpose8rS32_reversed_diagonale(&A[a_index / 8], &B[b_index / 8], lda / 8, ldb / 8);
            }
            //else if (j < m) {
            //    for (; j < m; ++j) {
            //        if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
            //        else remove_bit(B, j*ldb + i);
            //    }
            //}
        }
    }
}



__device__ __host__ uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

__device__ uint32_t reverse_32_bit(uint32_t a)
{
    // __device__ ​ unsigned int __brev(unsigned int  x) // CUDA
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return __brev(a);
    //return (reverse_8_bit(a >> 24) << 0) |
    //    (reverse_8_bit(a >> 16) << 8) |
    //    (reverse_8_bit(a >> 8) << 16) |
    //    (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

__device__ void transpose32_optimized(uint32_t A[32]) {
    int j, k;
    unsigned m, t;

    //m = 0x0000FFFF;
    //for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
    //    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
    //        t = (A[k] ^ (A[k + j] >> j)) & m;
    //        A[k] = A[k] ^ t;
    //        A[k + j] = A[k + j] ^ (t << j);
    //    }
    //}

    j = 16;
    m = 0x0000FFFF;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 8;
    m = 0x00ff00ff;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 4;
    m = 0x0f0f0f0f;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 2;
    m = 0x33333333;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 1;
    m = 0x55555555;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    // reverse Y
    for (j = 0; j < 16; ++j) {
        uint32_t tmp = A[j];
        A[j] = reverse_32_bit(A[31 - j]);
        A[31 - j] = reverse_32_bit(tmp);
    }
}

extern "C" {
__device__ void transpose_32x32_bits_reversed_diagonale(uint32_t *A, uint32_t *B, int m, int n)
{
    //unsigned A_tmp[32];
    //int i;
    //#pragma unroll
    //for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    //transpose32_optimized(A_tmp);
    //#pragma unroll
    //for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];

    __shared__ uint32_t A_shared[32 * BLOCK_TRANSPOSE32];
    uint32_t *A_tmp = &A_shared[32 * threadIdx.x];

    int i;
    #pragma unroll 32
    for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    transpose32_optimized(A_tmp);
    #pragma unroll 32
    for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];
}
}

// transpose 32x32 bit
__global__ void transpose_bin_gpu_kernel_32(uint32_t *A, uint32_t *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    int index = (blockIdx.x*blockDim.x + threadIdx.x) * 32;

    //for (i = 0; i < n; i += 8)
    {
        i = index % n;
        int j;
        //for (j = 0; j < m - 8; j += 8)
        {
            j = (index / n) * 32;
            if (j < m) {
                int a_index = i*lda + j;
                int b_index = j*ldb + i;
                transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
            }
        }
    }
}

void transpose_bin_gpu(unsigned char *A, unsigned char *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int size = n*m/ (8*8) + 1;
    int size32 = n*m / (32*32) + 1;
    const int num_blocks = size / BLOCK + 1;
    const int num_blocks32 = size32 / BLOCK_TRANSPOSE32 + 1;
    transpose_bin_gpu_kernel_32 << <num_blocks32, BLOCK_TRANSPOSE32, 0, get_cuda_stream() >> >((uint32_t *)A, (uint32_t *)B, n, m, lda, ldb, block_size);
    //transpose_bin_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(A, B, n, m, lda, ldb, block_size);
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------

__global__ void transpose_uint32_kernel(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
    //l.bit_align - algined (n) by 32
    //new_ldb - aligned (k) by 256
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    //for (i = 0; i < src_h; i += 1)
    int i = index % src_h;  // l.size*l.size*l.c;
    {
        //for (j = 0; j < src_w; j += 1)
        int j = index / src_h;  // out_h*out_w;
        if(j < src_w)
        {
            ((uint32_t *)dst)[j*dst_align / 32 + i] = ((uint32_t *)src)[i*src_align + j];
        }
    }
}

void transpose_uint32_gpu(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
    int size = src_w * src_h;
    const int num_blocks = size / BLOCK + 1;
    transpose_uint32_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(src, dst, src_h, src_w, src_align, dst_align);
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------

//#define TRANS_LOOP 10

__global__ void transpose_uint32_kernel_2(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
    __shared__ uint32_t tmp[33 * 32];   // misaligned_array[32x32]
    const int w_align = 33;
    //const int shared_size = w_align * 32;

    //l.bit_align - algined (n) by 32
    //new_ldb - aligned (k) by 256

    const int src_w_align = src_w + (32 - src_w % 32);
    const int src_h_align = src_h + (32 - src_h % 32);

    const int warps_in_width = src_w_align / 32;
    const int warps_in_height = src_h_align / 32;



    const int local_x = threadIdx.x % 32;   // index % 32;
    const int local_x_index = threadIdx.x / 32; // index / 32;
    const int local_y = local_x_index % 32;

//#pragma unroll TRANS_LOOP
    //for (int i = 0; i < TRANS_LOOP; ++i)
    {
        const int global_index = blockIdx.x;// blockIdx.x*TRANS_LOOP + i;// local_x_index / 32;
        const int global_x_index = global_index % warps_in_width;
        const int global_y_index = global_index / warps_in_width;

        const int global_x = global_x_index * 32 + local_x;
        const int global_y = global_y_index * 32 + local_y;

        uint32_t val = 0;
        if (global_x < src_w && global_y < src_h) {
            val = src[global_y * src_align + global_x];
        }
        //dst[global_x * dst_align / 32 + global_y] = val;
        //tmp[local_y * 32 + local_x] = val;

        tmp[local_x * w_align + local_y] = val;
        __syncthreads();
        val = tmp[local_y * w_align + local_x];

        const int new_global_x = global_y_index * 32 + local_x;
        const int new_global_y = global_x_index * 32 + local_y;

        if (new_global_x < src_h && new_global_y < src_w) {
            dst[new_global_y * (dst_align / 32) + new_global_x] = val;
        }
    }
}

#define TRANS_BLOCK 1024
void transpose_uint32_gpu_2(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
    int src_w_align = src_w + (32 - src_w % 32);
    int src_h_align = src_h + (32 - src_h % 32);

    int size = src_w_align * src_h_align;
    int num_blocks = size / TRANS_BLOCK;
    transpose_uint32_kernel_2 << <num_blocks, TRANS_BLOCK, 0, get_cuda_stream() >> >(src, dst, src_h, src_w, src_align, dst_align);
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
__global__ void repack_input_kernel(float *input, float *re_packed_input, int w, int h, int c)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    const int items_per_channel = w * h;

    int c_pack = index % 32;
    int chan_index = index / 32;
    int chan = (chan_index * 32) % c;
    int i = (chan_index * 32) / c;

    //for (chan = 0; chan < c; chan += 32)
    {
        //for (i = 0; i < items_per_channel; ++i)
        if(i < items_per_channel)
        {
            //for (c_pack = 0; c_pack < 32; ++c_pack)
            {
                float src = input[(chan + c_pack)*items_per_channel + i];

                re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
            }
        }
    }
}

void repack_input_gpu(float *input, float *re_packed_input, int w, int h, int c)
{
    int size = w * h * c;
    const int num_blocks = size / BLOCK + 1;
    repack_input_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(input, re_packed_input, w, h, c);
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
__global__ void repack_input_kernel_2(float *input, float *re_packed_input, int w, int h, int c)
{
    __shared__ uint32_t tmp[33 * 32];  // 33x32 is misaligned 32 x 32 to avoid bank conflicts

    int index = blockIdx.x*blockDim.x + threadIdx.x;

    const int items_per_channel = w * h;

    int c_pack = index % 32;
    int chan_index = index / 32;
    int chan = (chan_index * 32) % c;
    int i = (chan_index * 32) / c;

    //for (chan = 0; chan < c; chan += 32)
    {
        //for (i = 0; i < items_per_channel; ++i)
        if (i < items_per_channel)
        {
            //for (c_pack = 0; c_pack < 32; ++c_pack)
            {
                float src = input[(chan + c_pack)*items_per_channel + i];

                re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
            }
        }
    }
}

void repack_input_gpu_2(float *input, float *re_packed_input, int w, int h, int c)
{
    int size = w * h * c;
    const int num_blocks = size / BLOCK + 1;
    repack_input_kernel_2 << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(input, re_packed_input, w, h, c);
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
__global__ void repack_input_kernel_bin(float *input, uint32_t *re_packed_input_bin, int w, int h, int c)
{
    //__shared__ uint32_t tmp[32];
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    const int global_warp_id = index / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int items_per_channel = w * h;
    const int items_per_channel_aligned = items_per_channel + WARP_SIZE - (items_per_channel % WARP_SIZE);

    int i = 32 * (global_warp_id % (items_per_channel_aligned / WARP_SIZE));
    int chan = 32 * (global_warp_id / (items_per_channel_aligned / WARP_SIZE));

    if (chan < c)
    {
        uint32_t result_bits = 0;

        for (int c_pack = 0; c_pack < 32; ++c_pack)
        {
            float src = 0;
            if ((i + lane_id) < items_per_channel) {
                src = input[(chan + c_pack)*items_per_channel + (i + lane_id)];
            }
            uint32_t bit_mask = __ballot_custom(src > 0);

            uint32_t cur_bit = (bit_mask >> lane_id) & uint32_t(1);

            result_bits |= (cur_bit << c_pack);
        }
        if ((i + lane_id) < items_per_channel) {
            re_packed_input_bin[chan*items_per_channel / 32 + (i + lane_id)] = result_bits;
        }
    }
}

void repack_input_gpu_bin(float *input, uint32_t *re_packed_input_bin, int w, int h, int c)
{
    int size = (w * h * c) / 32 + 1;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    //printf("\n num_blocks = %d, num_blocks/32 = %d,  block_size = %d \n", num_blocks, num_blocks / 32, block_size);
    repack_input_kernel_bin << <num_blocks, block_size, 0, get_cuda_stream() >> >(input, re_packed_input_bin, w, h, c);
    CHECK_CUDA(cudaPeekAtLastError());
}

/*
// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
__global__ void repack_input_kernel_bin(float *input, uint32_t *re_packed_input_bin, int w, int h, int c)
{
    //__shared__ uint32_t tmp[32];
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    //const int num_of_warps = blockDim.x / WARP_SIZE;
    //const int warp_id = threadIdx.x / WARP_SIZE;
    //const int lane_id = threadIdx.x % WARP_SIZE;

    const int items_per_channel = w * h;

    int c_pack = index % 32;
    int chan_index = index / 32;
    //int chan = (chan_index * 32) % c;
    //int i = (chan_index * 32) / c;

    int i = (chan_index) % items_per_channel;
    int chan = ((chan_index ) / items_per_channel)*32;


    //for (chan = 0; chan < c; chan += 32)
    if(chan < c)
    {
        //for (i = 0; i < items_per_channel; ++i)
        //if (i < items_per_channel)
        {
            //for (c_pack = 0; c_pack < 32; ++c_pack)
            {
                float src = input[(chan + c_pack)*items_per_channel + i];

                uint32_t bit_mask = __ballot_custom(src > 0);
                if (threadIdx.x % 32 == 0)
                    re_packed_input_bin[chan*items_per_channel / 32 + i] = bit_mask;
            }
        }
    }
}

void repack_input_gpu_bin(float *input, uint32_t *re_packed_input_bin, int w, int h, int c)
{
    int size = w * h * c;
    const int block_size = 256;// 128;
    const int num_blocks = get_number_of_blocks(size, block_size);
    printf("\n num_blocks = %d, num_blocks/32 = %d,  block_size = %d \n", num_blocks, num_blocks/32, block_size);
    repack_input_kernel_bin << <num_blocks, block_size, 0, get_cuda_stream() >> >(input, re_packed_input_bin, w, h, c);
    CHECK_CUDA(cudaPeekAtLastError());
}
*/



__global__ void fill_int8_gpu_kernel(unsigned char *src, unsigned char val, size_t size) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < size) src[index] = 0;
}

void fill_int8_gpu(unsigned char *src, unsigned char val, size_t size) {
    const int num_blocks = size / BLOCK + 1;
    fill_int8_gpu_kernel<<<num_blocks, BLOCK, 0, get_cuda_stream()>>>(src, val, size);
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------

//typedef unsigned long long int uint64_t;
//typedef unsigned int uint32_t;
//typedef unsigned char uint8_t;
//typedef char int8_t;

__device__ __host__ static inline uint64_t broadcast_bit_1_to_64(uint8_t src) {
    return (src > 0) ? 0xFFFFFFFFFFFFFFFF : 0;
}

__device__ __host__ static inline uint8_t xnor_bit1(uint8_t a, uint8_t b) {
    return ~(a^b) & 0b1;
}

__device__ __host__ static inline uint32_t xnor_int32(uint32_t a, uint32_t b) {
    return ~(a^b);
}

__device__ __host__ static inline uint64_t xnor_int64(uint64_t a, uint64_t b) {
    return ~(a^b);
}

__device__ __host__ static inline uint4 xnor_int128(uint4 a, uint4 b) {
    uint4 res;
    res.w = ~(a.w^b.w);
    res.x = ~(a.x^b.x);
    res.y = ~(a.y^b.y);
    res.z = ~(a.z^b.z);
    return res;
}

__device__ __host__ static inline ulonglong4 xnor_int256(ulonglong4 a, ulonglong4 b) {
    ulonglong4 res;
    res.w = ~(a.w^b.w);
    res.x = ~(a.x^b.x);
    res.y = ~(a.y^b.y);
    res.z = ~(a.z^b.z);
    return res;
}

//-------

__device__ __host__ static inline uint8_t xor_bit1(uint8_t a, uint8_t b) {
    return (a^b) & 0b1;
}

__device__ __host__ static inline uint32_t xor_int32(uint32_t a, uint32_t b) {
    return (a^b);
}

__device__ __host__ static inline uint64_t xor_int64(uint64_t a, uint64_t b) {
    return (a^b);
}

__device__ __host__ static inline uint4 xor_int128(uint4 a, uint4 b) {
    uint4 res;
    res.w = (a.w^b.w);
    res.x = (a.x^b.x);
    res.y = (a.y^b.y);
    res.z = (a.z^b.z);
    return res;
}

__device__ __host__ static inline ulonglong4 xor_int256(ulonglong4 a, ulonglong4 b) {
    ulonglong4 res;
    res.w = (a.w^b.w);
    res.x = (a.x^b.x);
    res.y = (a.y^b.y);
    res.z = (a.z^b.z);
    return res;
}


__device__ static inline int popcnt_256(ulonglong4 a) {
    return __popcll(a.w) + __popcll(a.x) + __popcll(a.y) + __popcll(a.z);
}

/*
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    //if (index == 0)
    {
        int i, j, k, h;

        //#pragma omp parallel for
        //for (i = 0; i < M; ++i)
        i = index % M;
        //if(i < M)
        {   // l.n - filters [16 - 55 - 1024]
            float mean_val = mean_arr[i];

            //for (j = 0; j < N; ++j)
            j = index / M;
            if(j < N)
            { // out_h*out_w - one channel output size [169 - 173056]
                int count = 0;

                for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                    uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));
                    uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));
                    uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                    int tmp_count = __popcll(c_bit64);

                    if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                    count += tmp_count;
                    //binary_int64_printf(c_bit64);
                    //printf(", count = %d \n\n", tmp_count);
                }

                C[i*ldc + j] = (2 * count - K) * mean_val;
            }
        }
    }
}
*/


/*
// B (input) in the shared_memory
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{

    __shared__ uint64_t B_s[4096];  // 32 KB // [ldb x N`] // max = 262 144 bits

    int start_j = blockIdx.x*blockDim.x / M;
    {
        int end_j = (blockIdx.x*blockDim.x + blockDim.x) / M + 1;

        size_t shared_size = ldb * (end_j - start_j);

        //float tmp_shared_size = ldb * (blockDim.x / M);
        //int passes = (4096 * 64) / tmp_shared_size - 1;
        //size_t shared_size = tmp_shared_size * passes;

        int k;
        for (int k = threadIdx.x * 256; k < shared_size; k += blockDim.x * 256) {
            int x = start_j*ldb + k;
            if (x < (N*ldb)) *((ulonglong4 *)(B_s + k / 8)) = *((ulonglong4 *)(B + x / 8));
        }

        ////if (j_cur < N && (index % M == 0 || threadIdx.x == 0)) {
          ////  for (int k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
            ////    *((uint64_t *)(B_s + (local_j*ldb + k) / 8)) = *((uint64_t *)(B + (j_cur*ldb + k) / 8));    // input
            ////}
        ////}
    }
    __syncthreads();

    int index = blockIdx.x*blockDim.x + threadIdx.x;


    //if (index == 0)
    //for(int in_tmp = threadIdx.x; in_tmp < 1*blockDim.x; in_tmp += blockDim.x)
    {
        //int index = blockIdx.x*blockDim.x*1 + in_tmp;

        int j_cur = index / M;
        int local_j = j_cur - start_j;

        int i, j, h;

        //#pragma omp parallel for
        //for (i = 0; i < M; ++i)
        i = index % M;
        //if(i < M)
        {   // l.n - filters [16 - 55 - 1024]
            // further improvements: for (l.n == 1024) iterate several (j)
            float mean_val = mean_arr[i];

            //for (j = 0; j < N; ++j)
            j = index / M;
            if (j < N)
            { // out_h*out_w - one channel output size [169 - 173056]
                const int bit_step = 256;
                int count = 0;
                int k = 0;
                for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                    ulonglong4 a_bit256 = *((ulonglong4 *)(A + (i*lda + k) / 8));    // weights
                    //ulonglong4 b_bit256 = *((ulonglong4 *)(B + (j*ldb + k) / 8));
                    ulonglong4 b_bit256 = *((ulonglong4 *)(B_s + (local_j*ldb + k) / 8));    // input
                    ulonglong4 c_bit256 = xnor_int256(a_bit256, b_bit256);

                    count += __popcll(c_bit256.w) + __popcll(c_bit256.x) +
                        __popcll(c_bit256.y) + __popcll(c_bit256.z);
                }

                int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                //C[i*ldc + j] += 2 * count*mean_val;
                //C[i*ldc + j] += -2 * f1*mean_val;
                //C[i*ldc + j] += - K*mean_val;

                count = count - f1;    // remove extra bits (from empty space for align only)
                C[i*ldc + j] = (2 * count - K) * mean_val;

                //B_s[0] = (2 * count - K) * mean_val;
            }
        }
    }
}
*/

/*
// A (weights) in the shared_memory
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ uint64_t A_s[6144];  // 48 KB // [lda x M`]
                                    //__shared__ uint8_t A_s[6144*8];  // 48 KB // [lda x M`]

    int start_i = blockIdx.x*blockDim.x / N;
    int end_i = (blockIdx.x*blockDim.x + blockDim.x) / N + 1;

    size_t shared_size = lda * (end_i - start_i);

    int i_cur = index / N;
    int local_i = i_cur - start_i;

    for (int k = threadIdx.x * 64; k < shared_size; k += blockDim.x * 64) {
        int x = start_i*lda + k;
        if (x < (M*lda)) *((uint64_t *)(A_s + k / 8)) = *((uint64_t *)(A + x / 8));
    }

    //if (i_cur < M && (index % N == 0 || threadIdx.x == 0)) {
    //for (int k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
    //*((uint64_t *)(A_s + (local_i*lda + k) / 8)) = *((uint64_t *)(A + (i_cur*lda + k) / 8));    // weights
    //  }
    //}

    __syncthreads();

    int i, j, k, h;

    j = index % N;
    {    // out_h*out_w - one channel output size [169 - 173056]
        i = index / N;
        if (i < M)  // l.n - filters [16 - 55 - 1024]
        {
            float mean_val = mean_arr[i];
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                //uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
                uint64_t a_bit64 = *((uint64_t *)(A_s + (local_i*lda + k) / 8));    // weights
                uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                int tmp_count = __popcll(c_bit64);

                if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                count += tmp_count;
            }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}
*/

__inline__ __device__
int warpAllReduceSum(int val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
#if CUDART_VERSION >= 9000
        val += __shfl_xor_sync(FULL_MASK, val, mask);
#else
        val += __shfl_xor(val, mask);
#endif

    return val;
}

// Tensor Cores binary (CC >= 7.3 && CUDA >= 10.0) - __CUDA_SUBBYTE_IMMA__
#if CUDART_VERSION >= 10000
#include <mma.h>

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 128
#define WMMA_K32 (WMMA_K/32)

#define WMMA_Nx2 (WMMA_N*2)

// Tensor Cores are used for XOR-GEMM
__global__ void gemm_nn_custom_bin_mean_transposed_tensor_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr, int leaky_activation,
    float *shortcut_in_gpu, float *shortcut_out_gpu)
{
    // total 57%
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int C_s[WMMA_N * WMMA_M * 32 * 2];    // 2 * 8 KB - Temprorary result of GEMM WMMA for 32 warps

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int global_warp_id = index / 32;

    const int N_aligned = N + WMMA_Nx2 - (N % WMMA_Nx2);

    /*
    __syncthreads();
    __shared__ uint32_t A_s[8 * 512];   // 8x512 = 8 x 16384 bits, instead of 8x4
    const int start_global_warp_id = blockIdx.x*blockDim.x / 32;
    int start_i = start_global_warp_id / (N_aligned / WMMA_N);
    start_i = start_i * WMMA_M;
    if (start_i + WMMA_M > M) start_i = M - WMMA_M;   // must be: i+7 < M
    for (int tmp_index = threadIdx.x; tmp_index < (8 * 512); tmp_index += blockDim.x)
    {
        int k_tmp = tmp_index % 512;
        int local_i = tmp_index / 512;

        uint32_t a_val = ((uint32_t *)(A))[(start_i + local_i)*lda/32 + k_tmp];
        A_s[local_i * 512 + k_tmp] = a_val;
    }
    __syncthreads();
    */


    int i, j, k, h;
    // 47% = 29 + 10 + 8
    j = global_warp_id % (N_aligned / WMMA_Nx2);
    j = j * WMMA_Nx2;
    {    // out_h*out_w - one channel output size [169 - 173056]
        i = global_warp_id / (N_aligned / WMMA_Nx2);
        i = i * WMMA_M;

        int count = 0;
        k = 0;

        if (i < M)  //if (i < M)  // l.n - filters [16 - 55 - 1024]
        {
            if (j + WMMA_Nx2 > N) j = N - WMMA_Nx2;   // must be: j+7 < N
            if (i + WMMA_M > M) i = M - WMMA_M;   // must be: i+7 < M

#if __CUDA_ARCH__ >= 730
            // Tensor Cores
            using namespace nvcuda;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c1_frag, c2_frag;
            wmma::fill_fragment(c1_frag, 0); // !!!! XOR isn't XNOR !!!!!!!!!!
            wmma::fill_fragment(c2_frag, 0); // !!!! XOR isn't XNOR !!!!!!!!!!

            // 8 x 8 x 4 (uint32_t, 4 * 32 = 128 bit)
            for (; k < K; k += 128)  // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
            {
                int64_t A_cur_index = (i*lda + k) / 8;  // index in bits
                int64_t B1_cur_index = (j*ldb + k) / 8;  // index in bits
                int64_t B2_cur_index = ((j + 8)*ldb + k) / 8;  // index in bits

                // try to use A that is cached in shared memory - poor performance
                //if (i == start_i) wmma::load_matrix_sync(a_frag, &A_s[k / 32], (512 * 32));   // lda = (128*32) bits
                //else wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);   // lda = M

                // lda, ldb - are in bits
                wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);   // lda = M

                wmma::load_matrix_sync(b_frag, (uint32_t *)(B + B1_cur_index), ldb);   // ldb = K
                wmma::bmma_sync(c1_frag, a_frag, b_frag, c1_frag);    // XOR-GEMM

                wmma::load_matrix_sync(b_frag, (uint32_t *)(B + B2_cur_index), ldb);   // ldb = K
                wmma::bmma_sync(c2_frag, a_frag, b_frag, c2_frag);    // XOR-GEMM
            }
            // C[i*ldc + j]
            wmma::store_matrix_sync(&C_s[warp_id*WMMA_M*WMMA_N], c1_frag, WMMA_N, wmma::mem_row_major);
            wmma::store_matrix_sync(&C_s[warp_id*WMMA_M*WMMA_N + WMMA_M*WMMA_N*32], c2_frag, WMMA_N, wmma::mem_row_major);
#else // __CUDA_ARCH__ >= 730

            // Custom XOR-GEMM
            int k_d = lane_id % 4;
            int i_d = lane_id / 4;
            int j_d = lane_id / 4;

            int32_t accum_c_val[8*2]; // wmma::fill_fragment(c_frag, 0);
            for (int local_j = 0; local_j < 8*2; ++local_j) {
                accum_c_val[local_j] = 0;
            }

            // 8 x 8 x 4 (uint32_t, 4 * 32 = 128 bit)
            for (; k < K; k += 128)  // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
            {
                int64_t A_cur_index = (i*lda + k) / 8;
                //int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;

                // lda, ldb - are in bits
                // 8*4 = 32
                // 8*8 = 64
                int k_d = lane_id % 4;
                int i_d = lane_id / 4;
                int j_d = lane_id / 4;
                uint32_t a_val = *(uint32_t *)(A + ((i + i_d)*lda + (k + k_d*32)) / 8); // wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);

                for (int c_x = 0; c_x < 2; c_x++)
                {
                    uint32_t b_val = *(uint32_t *)(B + ((c_x * 8 + j + j_d)*ldb + (k + k_d * 32)) / 8); // wmma::load_matrix_sync(b_frag, (uint32_t *)(B + B_cur_index), ldb);

                    // wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);
                    int32_t c_val[8];  // 8 x 32 threads = 256
                    #pragma UNROLL
                    for (int local_j = 0; local_j < 8; ++local_j)
                    {
                        uint32_t b_val_cur = __shfl_custom(b_val, local_j * 4 + k_d);
                        c_val[local_j] = __popc(xor_int32(a_val, b_val_cur));
                    }

                    #pragma UNROLL
                    for (int local_j = 0; local_j < 8; ++local_j)
                    {
                        #pragma UNROLL
                        for (int local_k = 0; local_k < 4; ++local_k) {
                            accum_c_val[local_j + c_x*8] += __shfl_custom(c_val[local_j], i_d * 4 + local_k);
                        }
                    }
                }
            }

            // only the first 8 threads (i) contain 8 good values each, in c_val[8] (j) = 8 x 8 =64
            // wmma::store_matrix_sync(&C_s[warp_id*WMMA_M*WMMA_N], c_frag, WMMA_N, wmma::mem_row_major);
            if (k_d == 0) {
                for (int c_x = 0; c_x < 2; c_x++)
                {
                    for (int local_j = 0; local_j < 8; ++local_j)
                    {
                        C_s[warp_id*WMMA_M*WMMA_N + i_d*WMMA_N + local_j + WMMA_M*WMMA_N*32 * c_x] = accum_c_val[local_j + c_x*8];
                    }
                }
            }
#endif // __CUDA_ARCH__ >= 730

            for(int c_x = 0; c_x < 2; c_x++)
            {
                int j_d = lane_id % WMMA_N;
                {
                    #pragma UNROLL
                    for (int i_d = lane_id / WMMA_N; i_d < WMMA_M; i_d += WMMA_M / 2)
                    {
                        int count = C_s[warp_id*WMMA_M*WMMA_N + i_d*WMMA_N + j_d + WMMA_M*WMMA_N*32*c_x];

                        const int bit_step = 128;
                        int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                        count = count - f1;    // remove extra bits (from empty space for align only)

                        count = (2 * count - K);

                        float mean_val = mean_arr[i + i_d];
                        float bias_val = bias_arr[i + i_d];
                        float dst_val = count *mean_val + bias_val;
                        if (leaky_activation)
                            dst_val = (dst_val >= 0) ? (dst_val) : (0.1f*dst_val);    // Leaky activation

                        size_t out_index = (i + i_d)*ldc + (c_x * 8 + j + j_d);
                        C[out_index] = dst_val;

                        if (shortcut_out_gpu) {
                            shortcut_out_gpu[out_index] = shortcut_in_gpu[out_index] + dst_val;
                        }
                    }

                }
            }
        }
    }
}
#endif  // CUDART_VERSION >= 10000

/*
// Tensor Cores are used for XOR-GEMM
__global__ void gemm_nn_custom_bin_mean_transposed_tensor_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr, int leaky_activation)
{
    // total 57%
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int C_s[8*8 * 32];    // Temprorary result of GEMM WMMA

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int global_warp_id = index / 32;

    const int N_aligned = N + WMMA_N - (N % WMMA_N);

    int i, j, k, h;
    // 47% = 29 + 10 + 8
    j = global_warp_id % (N_aligned / WMMA_N);
    j = j * WMMA_N;
    {    // out_h*out_w - one channel output size [169 - 173056]
        i = global_warp_id / (N_aligned / WMMA_N);
        i = i * WMMA_M;

        int count = 0;
        k = 0;

        if (i < M)  //if (i < M)  // l.n - filters [16 - 55 - 1024]
        {
            if (j + WMMA_N > N) j = N - WMMA_N;   // must be: j+7 < N
            if (i + WMMA_M > M) i = M - WMMA_M;   // must be: i+7 < M

#if __CUDA_ARCH__ >= 730
            // Tensor Cores
            using namespace nvcuda;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;
            wmma::fill_fragment(c_frag, 0); // !!!! XOR isn't XNOR !!!!!!!!!!

            // 8 x 8 x 4 (uint32_t, 4 * 32 = 128 bit)
            for (; k < K; k += 128)  // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
            {
                int64_t A_cur_index = (i*lda + k) / 8;
                //int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;

                // lda, ldb - are in bits
                wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);   // lda = M
                wmma::load_matrix_sync(b_frag, (uint32_t *)(B + B_cur_index), ldb);   // ldb = K

                wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);    // XOR-GEMM
            }
            // C[i*ldc + j]
            wmma::store_matrix_sync(&C_s[warp_id*WMMA_M*WMMA_N], c_frag, WMMA_N, wmma::mem_row_major);
#else // __CUDA_ARCH__ >= 730

            // Custom XOR-GEMM
            int k_d = lane_id % 4;
            int i_d = lane_id / 4;
            int j_d = lane_id / 4;

            int32_t accum_c_val[8]; // wmma::fill_fragment(c_frag, 0);
            for (int local_j = 0; local_j < 8; ++local_j) {
                accum_c_val[local_j] = 0;
            }

            // 8 x 8 x 4 (uint32_t, 4 * 32 = 128 bit)
            for (; k < K; k += 128)  // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
            {
                int64_t A_cur_index = (i*lda + k) / 8;
                //int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;

                // lda, ldb - are in bits
                // 8*4 = 32
                // 8*8 = 64
                int k_d = lane_id % 4;
                int i_d = lane_id / 4;
                int j_d = lane_id / 4;
                uint32_t a_val = *(uint32_t *)(A + ((i + i_d)*lda + (k + k_d*32)) / 8); // wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);
                uint32_t b_val = *(uint32_t *)(B + ((j + j_d)*ldb + (k + k_d*32)) / 8); // wmma::load_matrix_sync(b_frag, (uint32_t *)(B + B_cur_index), ldb);

                // wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);
                int32_t c_val[8];  // 8 x 32 threads = 256
                #pragma UNROLL
                for (int local_j = 0; local_j < 8; ++local_j)
                {
                    uint32_t b_val_cur = __shfl_custom(b_val, local_j *4 + k_d);
                    c_val[local_j] = __popc(xor_int32(a_val, b_val_cur));
                }

                #pragma UNROLL
                for (int local_j = 0; local_j < 8; ++local_j)
                {
                    #pragma UNROLL
                    for (int local_k = 0; local_k < 4; ++local_k) {
                        accum_c_val[local_j] += __shfl_custom(c_val[local_j], i_d * 4 + local_k);
                    }
                }
            }

            // only the first 8 threads (i) contain 8 good values each, in c_val[8] (j) = 8 x 8 =64
            // wmma::store_matrix_sync(&C_s[warp_id*WMMA_M*WMMA_N], c_frag, WMMA_N, wmma::mem_row_major);
            if (k_d == 0) {
                for (int local_j = 0; local_j < 8; ++local_j)
                {
                    C_s[warp_id*WMMA_M*WMMA_N + i_d*WMMA_N + local_j] = accum_c_val[local_j];
                }
            }
#endif // __CUDA_ARCH__ >= 730

            {
                int i_d = lane_id % WMMA_M;
                {

                    for (int j_d = lane_id / WMMA_M; j_d < WMMA_N; j_d += WMMA_N / 2)
                    {
                        int count = C_s[warp_id*WMMA_M*WMMA_N + i_d*WMMA_N + j_d];

                        const int bit_step = 128;
                        int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                        count = count - f1;    // remove extra bits (from empty space for align only)

                        count = (2 * count - K);

                        float mean_val = mean_arr[i + i_d];
                        float bias_val = bias_arr[i + i_d];
                        float dst_val = count *mean_val + bias_val;
                        if (leaky_activation)
                            dst_val = (dst_val > 0) ? (dst_val) : (0.1f*dst_val);    // Leaky activation

                        C[(i + i_d)*ldc + (j + j_d)] = dst_val;
                    }

                }
            }
        }
    }
}
*/


// Coalescing
// A (weights) in the shared_memory - GOOD
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr, int leaky_activation,
    float *shortcut_in_gpu, float *shortcut_out_gpu)
{
    // total 57%
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ uint8_t A_s[6144*8/4];
    //__shared__ uint64_t A_s[6144];  // 48 KB // [lda x M`]
    //__shared__ uint8_t A_s[6144*8];  // 48 KB // [lda x M`]

    int start_i = blockIdx.x*blockDim.x / N;
    int end_i = (blockIdx.x*blockDim.x + blockDim.x) / N + 1;

    size_t shared_size = lda * (end_i - start_i);

    int i_cur = index / N;
    int local_i = i_cur - start_i;
    // ~10%
    for (int k = threadIdx.x * 64; k < shared_size; k += blockDim.x * 64) {
        int x = start_i*lda + k;
        if (x < (M*lda)) *((uint64_t *)(A_s + k / 8)) = *((uint64_t *)(A + x / 8));
    }
    __syncthreads();

    int i, j, k, h;
    // 47% = 29 + 10 + 8
    j = index % N;
    {    // out_h*out_w - one channel output size [169 - 173056]
        i = index / N;
        //if (i < M)  // l.n - filters [16 - 55 - 1024]
        {
            int count = 0;
            k = 0;

#ifdef NOT_USED
            // 32 thread X 256 bit = 8192 bit
            for (; k < (K - 8192); k += 8192) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                ulonglong4 c_bit256;

                //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

#pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl_custom(A_cur_index, t) + 32 * lane_id;
                    const int64_t B_i = __shfl_custom(B_cur_index, t) + 32 * lane_id;

                    {
                        //ulonglong4 a_bit256 = *((ulonglong4 *)(A + A_i));    // weights
                        ulonglong4 a_bit256 = *((ulonglong4 *)(A_s + A_i));    // weights
                        ulonglong4 b_bit256 = *((ulonglong4 *)(B + B_i));    // input
                        c_bit256 = xor_int256(a_bit256, b_bit256);
                        int tmp_count = __popcll(c_bit256.w) + __popcll(c_bit256.x) +
                            __popcll(c_bit256.y) + __popcll(c_bit256.z);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
#endif


//#ifdef NOT_USED
            // 32 thread X 64 bit = 2048 bit // 29%
            for (; k < (K - 2048); k += 2048) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t c_bit64;

                //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl_custom(A_cur_index, t) + 8 * lane_id;
                    const int64_t B_i = __shfl_custom(B_cur_index, t) + 8 * lane_id;

                    {
                        //uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
                        uint64_t a_bit64 = *((uint64_t *)(A_s + A_i));    // weights
                        uint64_t b_bit64 = *((uint64_t *)(B + B_i));    // input
                        c_bit64 = xor_int64(a_bit64, b_bit64);
                        int tmp_count = __popcll(c_bit64);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
//#endif

//#ifdef NOT_USED
            // 32 thread X 32 bit = 1024 bit // 10%
            for (; k < (K - 1024); k += 1024) {   // l.size*l.size*l.c - one filter size [27 - 9216]

                //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl_custom(A_cur_index, t) + 4 * lane_id;
                    const int64_t B_i = __shfl_custom(B_cur_index, t) + 4 * lane_id;

                    {
                        //uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
                        uint32_t a_bit32 = *((uint32_t *)(A_s + A_i));    // weights
                        uint32_t b_bit32 = *((uint32_t *)(B + B_i));    // input
                        uint32_t c_bit32 = xor_int32(a_bit32, b_bit32);
                        int tmp_count = __popc(c_bit32);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
//#endif

            if (i < M)
            {
                float mean_val = mean_arr[i];
                float bias_val = bias_arr[i];

//#ifdef NOT_USED
                // 8%
                for (; k < K; k += 256) {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                    //ulonglong4 a_bit256 = *((ulonglong4 *)(A + (i*lda + k) / 8));    // weights
                    ulonglong4 a_bit256 = *((ulonglong4 *)(A_s + (local_i*lda + k) / 8));    // weights
                    ulonglong4 b_bit256 = *((ulonglong4 *)(B + (j*ldb + k) / 8));    // input
                    ulonglong4 c_bit256 = xor_int256(a_bit256, b_bit256);

                    count += __popcll(c_bit256.w) + __popcll(c_bit256.x) +
                        __popcll(c_bit256.y) + __popcll(c_bit256.z);
                }
//#endif

#ifdef NOT_USED
                for (; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                    //uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
                    uint64_t a_bit64 = *((uint64_t *)(A_s + (local_i*lda + k) / 8));    // weights
                    uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
                    uint64_t c_bit64 = xor_int64(a_bit64, b_bit64);

                    count += __popcll(c_bit64);
                }
#endif

                const int bit_step = 256;
                int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    // remove extra bits (from empty space for align only)
                float dst_val = (2 * count - K) *mean_val + bias_val;
                if(leaky_activation)
                    dst_val = (dst_val >= 0) ? (dst_val) : (0.1f*dst_val);    // Leaky activation
                size_t out_index = i*ldc + j;
                C[out_index] = dst_val;

                if (shortcut_out_gpu) {
                    shortcut_out_gpu[out_index] = shortcut_in_gpu[out_index] + dst_val;
                }
            }
        }
    }
}


// further optimization - use WMMA GEMM for using Tensor Cores
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/cudaTensorCoreGemm/cudaTensorCoreGemm.cu
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-subbyte
// nvcuda::wmma::col_major ->  cutlass::MatrixLayout::kColumnMajor (matrix is not transposed)

// Matrix A	Matrix B	Accumulator	Matrix Size (m-n-k)
// precision::b1	precision::b1	int	8x8x128

// The only dimensions currently supported by WMMA for XNOR
// const int WMMA_M = 8;
// const int WMMA_N = 8;
// const int WMMA_K = 128;


// GOOD
void gemm_nn_custom_bin_mean_transposed_gpu(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias, int leaky_activation,
    float *shortcut_in_gpu, float *shortcut_out_gpu)
{
    int size = M*N;
    const int num_blocks = get_number_of_blocks(size, BLOCK);

    //printf("\n M = %d, N = %d, M %% 8 = %d, N %% 8 = %d \n", M, N, M % 8, N % 8);

    /*
    printf("\n gemm_bin size = %d, num_blocks = %d, M*K = %d KB, N*K = %d KB \n (w) M*K/num_blocks = %d KB, (i) N*K/num_blocks = %d KB \n",
        size, num_blocks, M*K / 1024, N*K / 1024, M*lda / num_blocks / 1024, N*ldb / num_blocks / 1024);
    printf(" M / 512 = %d, N / 512 = %d, M*lda / 512 = %d, N*ldb / 512 = %d \n", M / 512, N / 512, M*lda/512, N*ldb/512);
    */
    //printf(" shared_memory: (w) lda*BLOCK/N = %d, (i) ldb*BLOCK/M = %d, \t lda = %d \n\n", lda*BLOCK / N, ldb*BLOCK / M, lda);


    //if (M % 8 == 0 && N % 8 == 0 && M == 128)
    //if (M >= 32)    // l.n >= 32
#if CUDART_VERSION >= 10000
    if (1)
    {
        const int M_aligned = M + (8 - (M % 8));
        const int N_aligned = N + (16 - (N % 16));
        int size = (M_aligned / 8)*(N_aligned / 16)*WARP_SIZE;
        const int num_blocks = get_number_of_blocks(size, BLOCK);

        //printf(" lda = %d, ldb = %d, ldc = %d, lda/32 = %d, ldb/32 = %d, ldc/32 = %d \n", lda, ldb, ldc, lda / 32, ldb / 32, ldc / 32);
        //printf("  l.c (K/9) = %d, M (l.n) = %d \n", (K%9 == 0)? K / 9: K, M);
        gemm_nn_custom_bin_mean_transposed_tensor_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (
            M, N, K,
            A, lda,
            B, ldb,
            C, ldc,
            mean_arr, bias, leaky_activation,
            shortcut_in_gpu, shortcut_out_gpu);

        //cudaDeviceSynchronize();
        //getchar();
    }
    else
#endif  //# CUDART_VERSION >= 10000
    {
        gemm_nn_custom_bin_mean_transposed_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (
            M, N, K,
            A, lda,
            B, ldb,
            C, ldc,
            mean_arr, bias, leaky_activation,
            shortcut_in_gpu, shortcut_out_gpu);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------


void convolve_cpu(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n, int size, int pad)
{
    int fil;
    // filter index
#pragma omp parallel for      // "omp parallel for" - automatic parallelization of loop by using OpenMP
    for (fil = 0; fil < n; ++fil) {
        int chan, y, x, f_y, f_x;
        // channel index
        for (chan = 0; chan < in_c; ++chan)
            // input - y
            for (y = 0; y < in_h; ++y)
                // input - x
                for (x = 0; x < in_w; ++x)
                {
                    int const output_index = fil*in_w*in_h + y*in_w + x;
                    int const weights_pre_index = fil*in_c*size*size + chan*size*size;
                    int const input_pre_index = chan*in_w*in_h;
                    float sum = 0;

                    // filter - y
                    for (f_y = 0; f_y < size; ++f_y)
                    {
                        int input_y = y + f_y - pad;
                        // filter - x
                        for (f_x = 0; f_x < size; ++f_x)
                        {
                            int input_x = x + f_x - pad;
                            if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

                            int input_index = input_pre_index + input_y*in_w + input_x;
                            int weights_index = weights_pre_index + f_y*size + f_x;

                            sum += input[input_index] * weights[weights_index];
                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    output[output_index] += sum;
                }
    }


}
// --------------------------------


void convolve_bin_cpu(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n,
    int size, int pad, int new_lda, float *mean_arr_gpu)
{
    int fil;
    // filter index
#pragma omp parallel for      // "omp parallel for" - automatic parallelization of loop by using OpenMP
    for (fil = 0; fil < n; ++fil) {
        float mean_val = mean_arr_gpu[fil];
        int chan, y, x, f_y, f_x;
        // channel index
        for (chan = 0; chan < in_c; ++chan)
            // input - y
            for (y = 0; y < in_h; ++y)
                // input - x
                for (x = 0; x < in_w; ++x)
                {
                    int const output_index = fil*in_w*in_h + y*in_w + x;
                    int const weights_pre_index = fil*in_c*size*size + chan*size*size;
                    int const input_pre_index = chan*in_w*in_h;
                    int sum = 0;
                    int good_val = 0;

                    // filter - y
                    for (f_y = 0; f_y < size; ++f_y)
                    {
                        int input_y = y + f_y - pad;
                        // filter - x
                        for (f_x = 0; f_x < size; ++f_x)
                        {
                            int input_x = x + f_x - pad;
                            if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

                            int input_index = input_pre_index + input_y*in_w + input_x;
                            //int weights_index = weights_pre_index + f_y*size + f_x;
                            //int weights_index = fil*in_c*size*size + chan*size*size + f_y*size + f_x;
                            int weights_index = fil*new_lda + chan*size*size + f_y*size + f_x;

                            //sum += input[input_index] * weights[weights_index];

                            int8_t in_bit = get_bit((uint8_t *)input, input_index);
                            int8_t w_bit = get_bit((uint8_t *)weights, weights_index);
                            int res = xnor_bit1(in_bit, w_bit);
                            sum += res;
                            good_val++;
                            //sum += (res > 0) ? 1 : -1;
                            //in_bit = (in_bit > 0) ? 1 : -1;
                            //w_bit = (w_bit > 0) ? 1 : -1;
                            //int8_t res = in_bit*w_bit;
                            //sum += res;
                            //printf("\n i: %d x w: %d = res: %d \t sum: %d \t mean = %f \n", in_bit, w_bit, res, sum, mean_val);
                        }
                    }
                    //printf("sum = %d, ", sum);
                    sum = sum - (good_val - sum);
                    //printf(" size = %d, sum = %d \n", size, sum);

                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    output[output_index] += sum*mean_val;
                }
    }
}
// --------------------------------

__global__ void convolve_gpu_kernel(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n, int size, int pad)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    int fil;
    // filter index
    //for (fil = 0; fil < n; ++fil)
    int chan, y, x, f_y, f_x;
    // channel index
    //for (chan = 0; chan < in_c; ++chan)
    // input - y
    //for (y = 0; y < in_h; ++y)
    // input - x
    //for (x = 0; x < in_w; ++x)
    x = index % in_w;
    int index2 = index / in_w;
    y = index2 % in_h;
    fil = index2 / in_h;
    if (fil < n)
    {

        int const output_index = fil*in_w*in_h + y*in_w + x;
        float sum = 0;

        for (chan = 0; chan < in_c; ++chan)
        {
            int const weights_pre_index = fil*in_c*size*size + chan*size*size;
            int const input_pre_index = chan*in_w*in_h;

            // filter - y
            for (f_y = 0; f_y < size; ++f_y)
            {
                int input_y = y + f_y - pad;
                // filter - x
                for (f_x = 0; f_x < size; ++f_x)
                {
                    int input_x = x + f_x - pad;
                    if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

                    int input_index = input_pre_index + input_y*in_w + input_x;
                    int weights_index = weights_pre_index + f_y*size + f_x;

                    sum += input[input_index] * weights[weights_index];

                }
            }
            // l.output[filters][width][height] +=
            //        state.input[channels][width][height] *
            //        l.weights[filters][channels][filter_width][filter_height];
            //output[output_index] += sum;
        }
        output[output_index] = sum;
    }

}

void convolve_gpu(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n, int size, int pad)
{
    int array_size = in_w*in_h*n;    // width X height X filters
    const int num_blocks = array_size / BLOCK + 1;
    //printf("\n array_size = %d, num_blocks = %d, w = %d, h = %d, n = %d, c = %d, pad = %d \n", array_size, num_blocks, in_w, in_h, n, in_c, pad);

    convolve_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (input, weights, output, in_w, in_h, in_c, n, size, pad);
    CHECK_CUDA(cudaPeekAtLastError());
}

// --------------------------------

/*
__global__ void convolve_bin_gpu_kernel(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n,
    int size, int pad, int new_lda, float *mean_arr_gpu)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    int fil;
    // filter index
    //for (fil = 0; fil < n; ++fil)
    int chan, y, x, f_y, f_x;
    // channel index
    //for (chan = 0; chan < in_c; ++chan)
    // input - y
    //for (y = 0; y < in_h; ++y)
    // input - x
    //for (x = 0; x < in_w; ++x)
    x = index % in_w;
    int index2 = index / in_w;
    y = index2 % in_h;
    fil = index2 / in_h;
    if (fil < n)    // (1-6 for one BLOCK)
    {
                //float mean_val = mean_arr_gpu[fil];
                int const output_index = fil*in_w*in_h + y*in_w + x;
                int sum = 0;
                int good_val = 0;

                for (chan = 0; chan < in_c; ++chan)
                {
                    //int const weights_pre_index = fil*in_c*size*size + chan*size*size;
                    int const weights_pre_index = fil*new_lda + chan*size*size;
                    int const input_pre_index = chan*in_w*in_h;

                    // filter - y
                    for (f_y = 0; f_y < size; ++f_y)
                    {
                        int input_y = y + f_y - pad;
                        // filter - x
                        for (f_x = 0; f_x < size; ++f_x)
                        {
                            int input_x = x + f_x - pad;
                            if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

                            int input_index = input_pre_index + input_y*in_w + input_x;
                            int weights_index = weights_pre_index + f_y*size + f_x;
                            //int weights_index = fil*in_c*size*size + chan*size*size + f_y*size + f_x;
                            //int weights_index = fil*new_lda + chan*size*size + f_y*size + f_x;

                            uint8_t in_bit = get_bit((uint8_t *)input, input_index);
                            uint8_t w_bit = get_bit((uint8_t *)weights, weights_index);
                            int res = xnor_bit1(in_bit, w_bit);
                            sum += res;
                            good_val++;

                            //sum += input[input_index] *weights[weights_index];

                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    //output[output_index] += sum;
                }
                sum = sum - (good_val - sum);
                output[output_index] = sum * mean_arr_gpu[fil]; // atoimcAdd for inter-BLOCK sum
    }

}
*/

__global__ void convolve_bin_gpu_kernel(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n,
    int size, int pad, int new_lda, float *mean_arr_gpu)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    int fil;
    // filter index
    //for (fil = 0; fil < n; ++fil)
    int chan, y, x, f_y, f_x;
    // channel index
    //for (chan = 0; chan < in_c; ++chan)
    // input - y
    //for (y = 0; y < in_h; ++y)
    // input - x
    //for (x = 0; x < in_w; ++x)
    x = index % in_w;
    int index2 = index / in_w;
    y = index2 % in_h;
    fil = index2 / in_h;
    //if (fil < n)    // (1-6 for one BLOCK)
    {
        //float mean_val = mean_arr_gpu[fil];
        int const output_index = fil*in_w*in_h + y*in_w + x;
        int sum = 0;
        int good_val = 0;

        int min_index = blockIdx.x*blockDim.x;
        int min_fil = (min_index / in_w) / in_h;
        int max_index = (blockIdx.x+1)*blockDim.x - 1;
        int max_fil = (max_index / in_w) / in_h;

        __shared__ uint32_t weights_shared[3*3*1024*6/32 + 1];  // 7 KB (6 filters) - use (new_lda) for size calculation
        //const int weights_size = size*size*in_c/8;
        const int weights_size = size*size*in_c / 32 + 1;

        for (int tmp_fil = min_fil; tmp_fil <= max_fil; tmp_fil++) {
            for (int s = threadIdx.x; s < weights_size; s += blockDim.x) {
                //weights_shared[s + (tmp_fil - min_fil)*new_lda / 8] = ((uint8_t *)weights)[tmp_fil*new_lda / 8 + s];
                weights_shared[s + (tmp_fil - min_fil)*new_lda/32] = ((uint32_t *)weights)[tmp_fil*new_lda / 32 + s];
            }
        }
        __syncthreads();

        for (chan = 0; chan < in_c; ++chan)
        {
            //int const weights_pre_index = fil*in_c*size*size + chan*size*size;
            int const weights_pre_index = fil*new_lda + chan*size*size;
            int const input_pre_index = chan*in_w*in_h;

            __shared__ uint32_t input_shared[416*416/32 + 1];   // 21.2 KB bytes (for input size 832x832)
            const int input_shared_size = in_w*in_h / 32 + 1;
            const int add_input_index = input_pre_index % 32;
            __syncthreads();    // why??? but is required

            for (int s = threadIdx.x; s < input_shared_size; s += blockDim.x) {
                input_shared[s] = ((uint32_t *)input)[input_pre_index / 32 + s];
            }
            __syncthreads();

            /*
            __shared__ uint8_t input_shared[208 * 208 / 8 + 1];   // 5.4 KB bytes (for input size 416x416)
            const int input_shared_size = in_w*in_h / 8 + 1;
            const int add_input_index = input_pre_index % 8;
            __syncthreads();

            for (int s = threadIdx.x; s < input_shared_size; s += blockDim.x) {
                ((uint8_t *)input_shared)[s] = ((uint8_t *)input)[input_pre_index / 8 + s];
            }
            __syncthreads();
            */
            int src_index = -1;
            uint32_t input_byte;

            if (fil < n)    // (1-6 for one BLOCK)
            {
                // filter - y
                for (f_y = 0; f_y < size; ++f_y)
                {
                    int input_y = y + f_y - pad;
                    // filter - x
                    for (f_x = 0; f_x < size; ++f_x)
                    {
                        int input_x = x + f_x - pad;
                        if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

                        int input_index = input_pre_index + input_y*in_w + input_x;
                        int weights_index = weights_pre_index + f_y*size + f_x;
                        //int weights_index = fil*in_c*size*size + chan*size*size + f_y*size + f_x;
                        //int weights_index = fil*new_lda + chan*size*size + f_y*size + f_x;

                        //uint8_t in_bit = get_bit((uint8_t *)input, input_index);
                        //uint8_t w_bit = get_bit((uint8_t *)weights, weights_index);

                        //int weights_index = fil*in_c*size*size + chan*size*size + f_y*size + f_x;
                        int weights_shared_index = (fil - min_fil)*new_lda + chan*size*size + f_y*size + f_x;
                        //uint8_t in_bit = get_bit((uint8_t *)weights_shared, weights_shared_index);
                        uint8_t w_bit = get_bit((uint8_t *)weights_shared, weights_shared_index);

                        //int input_index = input_pre_index + input_y*in_w + input_x;
                        int input_shared_index = /*input_pre_index +*/ input_y*in_w + input_x + add_input_index;
                        uint8_t in_bit = get_bit((uint8_t *)input_shared, input_shared_index);
                        /*
                        int new_src_index = input_shared_index / 32;
                        int src_shift = input_shared_index % 32;
                        //if (new_src_index != src_index)
                        {
                            src_index = new_src_index;
                            input_byte = ((uint32_t *)input_shared)[src_index];
                        }
                        uint8_t in_bit = (input_byte & (1 << src_shift)) >> src_shift;
                        */

                        int res = xnor_bit1(in_bit, w_bit);
                        sum += res;
                        good_val++;

                        //sum += input[input_index] *weights[weights_index];

                    }
                }
            }
            // l.output[filters][width][height] +=
            //        state.input[channels][width][height] *
            //        l.weights[filters][channels][filter_width][filter_height];
            //output[output_index] += sum;
        }
        sum = sum - (good_val - sum);
        //output[output_index] = sum * mean_arr_gpu[fil]; // atoimcAdd for inter-BLOCK sum
        atomicAdd(&output[output_index], sum * mean_arr_gpu[fil]);
    }

}

void convolve_bin_gpu(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n,
    int size, int pad, int new_lda, float *mean_arr_gpu)
{
    int array_size = in_w*in_h*n;    // width X height X filters
    const int num_blocks = array_size / BLOCK + 1;
    //printf("\n array_size = %d, num_blocks = %d, w = %d, h = %d, n = %d, c = %d, pad = %d \n", array_size, num_blocks, in_w, in_h, n, in_c, pad);

    convolve_bin_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (input, weights, output, in_w, in_h, in_c, n, size, pad, new_lda, mean_arr_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}

// --------------------------------
