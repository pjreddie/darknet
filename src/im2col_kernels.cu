#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <stdint.h>

extern "C" {
#include "im2col.h"
#include "cuda.h"
}

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define WARP_SIZE 32


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

//#define SHRED_VALS ((BLOCK / 169) * )
    //__shared__ float dst_s[1024];
    //__shared__ float dst_s[1024];
    //__shared__ uint32_t bit_s[32];
    //__shared__ uint8_t bit_s[128];

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
}


// --------------------------------

/*
// binary im2col
__global__ void im2col_align_bin_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize, const int channels,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    float *data_col, const int bit_align)
{
    __shared__ float tmp_s[1];

    //#define SHRED_VALS ((BLOCK / 169) * )
    __shared__ float dst_s[1024];
    //__shared__ float dst_s[1024];
    //__shared__ uint32_t bit_s[32];
    __shared__ uint8_t bit_s[128];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x*gridDim.x)
    {
        //int c_index = index;
        //int channel_in = c_index % channels;

        int h_out = index % height_col;
        int c_index = index / height_col;
        int channel_in = c_index % channels;

        int channel_out = channel_in * ksize * ksize;

        int j_index = c_index / channels;
        int j = j_index % ksize;
        int i = j_index / ksize;
        if (i < ksize)
        {
            for (int w_out = 0; w_out < width_col; ++w_out)
            {
                int h_in = h_out * stride - pad;
                int w_in = w_out * stride - pad;

                int h = h_in + i;
                int w = w_in + j;

                float val = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im[(channel_in * height + h_in) * width + w_in + i * width + j] : 0;

                //int pre_out_index = index % (width_col*height_col);
                int pre_out_index = h_out * width_col + w_out;
                int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;
                data_col[out_index] = val;


            }// w_out
        }
    }
}
*/

/*
// binary im2col
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
        //int c_index = index;
        //int channel_in = c_index % channels;

        int h_out = index % height_col;
        int c_index = index / height_col;
        int channel_in = c_index % channels;

        int channel_out = channel_in * ksize * ksize;

        int j_index = c_index / channels;
        int j = j_index % ksize;
        int i = j_index / ksize;

        int h_in = h_out * stride - pad;
        int h = h_in + i;

        //if (i < ksize)
        {
            int w_out = 0;

            // the end of padding
            //if(0)
            for (; w_out < (width_col); w_out += 32)
            {
                int w = w_out * stride - pad + j;
                int pre_in_index = (channel_in * height + h_in) * width + i * width;
                int in_index = pre_in_index + w;
                //float *src_p = (float *)&data_im[in_index];

                int pre_out_index = h_out * width_col + w_out;
                int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;
                // float *dst_p = (float *)&data_col[out_index];

                if (i >= ksize) {
                    out_index = -1;
                }

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    //const int64_t cur_pre_in_index = pre_in_index;
                    //const int64_t cur_j = j;
                    //const int64_t out_i = out_index;// __shfl(out_index, t) + lane_id;

                    const int64_t cur_out_index = __shfl(out_index, t);
                    if (cur_out_index >= 0)
                    {
                        const int64_t cur_pre_in_index = __shfl(pre_in_index, t);
                        const int64_t cur_j = __shfl(j, t);
                        const int64_t cur_h = __shfl(h, t);

                        int cur_w = ((w_out + lane_id) * stride - pad + cur_j);
                        int in_index = cur_pre_in_index + cur_w;

                        float val = (cur_w >= 0 && cur_w < width && cur_h >= 0 && cur_h < height) ?
                            data_im[in_index] : float();

                        if ((w_out + lane_id) < width_col) {
                            data_col[cur_out_index + lane_id] = val;
                            //tmp_s[0] = val;

                            //uint32_t bit_mask = __ballot(val > 0);
                            //uint8_t *bit8_ptr = &(((uint8_t *)data_col)[cur_out_index / 8]);
                            //uint32_t *bit32_ptr = (uint32_t *)bit8_ptr;
                            //*bit32_ptr = bit_mask;
                        }
                    }
                }

            }// w_out

#ifdef NOT_USED
            if (i < ksize && h >= 0 && h < height)
            {

                // wait for align address and the end of padding
                for (; w_out < width_col; ++w_out)
                {
                    int w_in = w_out * stride - pad;
                    int w = w_in + j;

                    int in_index = (channel_in * height + h_in) * width + w_in + i * width + j;
                    float *src_p = (float *)&data_im[in_index];

                    int pre_out_index = h_out * width_col + w_out;
                    int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;
                    float *dst_p = (float *)&data_col[out_index];

                    if (((uint64_t)src_p % 32 == 0) && ((uint64_t)dst_p % 32 == 0) && w > 0) {
                        //printf(" aligned addresses and there is no padding \n");
                        break;
                    }

                    float val = (w >= 0 && w < width) ?
                        (*src_p) : float();

                    *dst_p = val;
                    //tmp_s[0] = val;
                }// w_out

                // ulonglong4 (256 bit) / instead of float (32 bit) = 8x times
                for (; w_out < (width_col - 8); w_out += 8)
                {
                    int w_in = w_out * stride - pad;
                    int w = w_in + j;

                    ulonglong4 *src_p = (ulonglong4 *)&data_im[(channel_in * height + h_in) * width + w_in + i * width + j];

                    int pre_out_index = h_out * width_col + w_out;
                    int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;
                    ulonglong4 *dst_p = (ulonglong4 *)&data_col[out_index];

                    ulonglong4 val = (w < width) ?
                        (*src_p) : ulonglong4();

                    *dst_p = val;
                    //tmp256_s[0] = val;
                }// w_out

                for (; w_out < width_col; ++w_out)
                {
                    //int h_in = h_out * stride - pad;
                    int w_in = w_out * stride - pad;

                    //int h = h_in + i;
                    int w = w_in + j;

                    float val = (w < width) ?
                        data_im[(channel_in * height + h_in) * width + w_in + i * width + j] : 0;

                    int pre_out_index = h_out * width_col + w_out;
                    int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;
                    data_col[out_index] = val;
                    //tmp_s[0] = val;
                }// w_out
            }
#endif  // NOT_USED
        }
    }
}
*/


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

                    const int cur_wh_index = __shfl(send_wh_index, t) + lane_id;

                    if (cur_wh_index < (width_col*height_col))// && (cur_i_pad+pad) < ksize)
                    {
                        const int cur_pre_out_index = __shfl(pre_out_index, t);

                        const int cur_pre_in_index = __shfl(pre_in_index, t);
                        const int cur_pre_in_wh_index = __shfl(pre_in_wh_index, t) + lane_id;

                        int w = cur_pre_in_wh_index % width;
                        int h = cur_pre_in_wh_index / width;
                        int in_index = cur_pre_in_index + cur_pre_in_wh_index;

                        int out_index = cur_pre_out_index + cur_wh_index;

                        float val = (w >= 0 && w < width && h >= 0 && h < height) ?
                            data_im[in_index] : float();

                        //data_col[out_index] = val;
                        //tmp_s[0] = val;

                        uint32_t bit_mask = __ballot(val > 0);
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
        unsigned int bit_mask = __ballot(src_val > 0);
        if (threadIdx.x % WARP_SIZE == 0) ((unsigned int*)dst)[index / 32] = bit_mask;
    }
}
*/

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

        uint32_t bit_mask = __ballot(src_val > 0);
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


void float_to_bit_gpu(float *src, unsigned char *dst, size_t size)
{
    const int num_blocks = size / 1024 + 1;
    float_to_bit_gpu_kernel<<<num_blocks, 1024, 0, get_cuda_stream()>>>(src, dst, size);
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

__device__ __host__ void transpose8rS32_reversed_diagonale(unsigned char* A, int m, int n, unsigned char* B)
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

    B[7 * n] = reverse_byte(x >> 24);  B[6 * n] = reverse_byte(x >> 16);  B[5 * n] = reverse_byte(x >> 8);  B[4 * n] = reverse_byte(x);
    B[3 * n] = reverse_byte(y >> 24);  B[2 * n] = reverse_byte(y >> 16);  B[1 * n] = reverse_byte(y >> 8);  B[0 * n] = reverse_byte(y);

    //__device__ ​ unsigned int 	__brev(unsigned int  x)
    //Reverse the bit order of a 32 bit unsigned integer.
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html
}


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
            if (j < m - 8) {
                int a_index = i*lda + j;
                int b_index = j*ldb + i;
                transpose8rS32_reversed_diagonale(&A[a_index / 8], lda / 8, ldb / 8, &B[b_index / 8]);
            }
            else if (j < m) {
                for (; j < m; ++j) {
                    if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
                    else remove_bit(B, j*ldb + i);
                }
            }
        }
    }
}


void transpose_bin_gpu(unsigned char *A, unsigned char *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    size_t size = n*m/64 + 1;
    const int num_blocks = size / BLOCK + 1;
    transpose_bin_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(A, B, n, m, lda, ldb, block_size);
}
// --------------------------------


__global__ void fill_int8_gpu_kernel(unsigned char *src, unsigned char val, size_t size) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < size) src[index] = 0;
}

void fill_int8_gpu(unsigned char *src, unsigned char val, size_t size) {
    const int num_blocks = size / BLOCK + 1;
    fill_int8_gpu_kernel<<<num_blocks, BLOCK, 0, get_cuda_stream()>>>(src, val, size);
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
        val += __shfl_xor(val, mask);
    return val;
}


// Coalescing
// A (weights) in the shared_memory - GOOD
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ uint8_t A_s[6144*8/4];
    //__shared__ uint64_t A_s[6144];  // 48 KB // [lda x M`]
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
    __syncthreads();

    int i, j, k, h;

    j = index % N;
    {    // out_h*out_w - one channel output size [169 - 173056]
        i = index / N;
        //if (i < M)  // l.n - filters [16 - 55 - 1024]
        {
            int count = 0;
            k = 0;

//#ifdef NOT_USED
            // 32 thread X 64 bit = 2048 bit
            for (; k < (K - 2048); k += 2048) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t c_bit64;

                //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl(A_cur_index, t) + 8 * lane_id;
                    const int64_t B_i = __shfl(B_cur_index, t) + 8 * lane_id;

                    {
                        //uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
                        uint64_t a_bit64 = *((uint64_t *)(A_s + A_i));    // weights
                        uint64_t b_bit64 = *((uint64_t *)(B + B_i));    // input
                        c_bit64 = xnor_int64(a_bit64, b_bit64);
                        int tmp_count = __popcll(c_bit64);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
//#endif

//#ifdef NOT_USED
            // 32 thread X 32 bit = 1024 bit
            for (; k < (K - 1024); k += 1024) {   // l.size*l.size*l.c - one filter size [27 - 9216]

                //int64_t A_cur_index = (i*lda + k) / 8;
                int64_t A_cur_index = (local_i*lda + k) / 8;
                int64_t B_cur_index = (j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl(A_cur_index, t) + 4 * lane_id;
                    const int64_t B_i = __shfl(B_cur_index, t) + 4 * lane_id;

                    {
                        //uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
                        uint32_t a_bit32 = *((uint32_t *)(A_s + A_i));    // weights
                        uint32_t b_bit32 = *((uint32_t *)(B + B_i));    // input
                        uint32_t c_bit32 = xnor_int32(a_bit32, b_bit32);
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
                for (; k < K; k += 256) {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                    //ulonglong4 a_bit256 = *((ulonglong4 *)(A + (i*lda + k) / 8));    // weights
                    ulonglong4 a_bit256 = *((ulonglong4 *)(A_s + (local_i*lda + k) / 8));    // weights
                    ulonglong4 b_bit256 = *((ulonglong4 *)(B + (j*ldb + k) / 8));    // input
                    ulonglong4 c_bit256 = xnor_int256(a_bit256, b_bit256);

                    count += __popcll(c_bit256.w) + __popcll(c_bit256.x) +
                        __popcll(c_bit256.y) + __popcll(c_bit256.z);
                }
//#endif

#ifdef NOT_USED
                for (; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                    //uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
                    uint64_t a_bit64 = *((uint64_t *)(A_s + (local_i*lda + k) / 8));    // weights
                    uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
                    uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                    count += __popcll(c_bit64);
                }
#endif

                const int bit_step = 256;
                int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    // remove extra bits (from empty space for align only)

                C[i*ldc + j] = (2 * count - K) *mean_val + bias_val;
            }
        }
    }
}


/*
// Coalescing
// B (input) in the shared_memory - GOOD
__global__ void gemm_nn_custom_bin_mean_transposed_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ uint8_t B_s[4096*8];  // 32 KB // [ldb x N`] // max = 262 144 bits
    //__shared__ uint64_t B_s[4096];  // 32 KB // [ldb x N`] // max = 262 144 bits

    int start_j = blockIdx.x*blockDim.x / M;
    int end_j = (blockIdx.x*blockDim.x + blockDim.x) / M + 1;

    size_t shared_size = ldb * (end_j - start_j);

    int j_cur = index / M;
    int local_j = j_cur - start_j;

    for (int k = threadIdx.x * 256; k < shared_size; k += blockDim.x * 256) {
        int x = start_j*ldb + k;
        if (x < (N*ldb)) *((ulonglong4 *)(B_s + k / 8)) = *((ulonglong4 *)(B + x / 8));
    }
    __syncthreads();

    int i, j, k;

    i = index % M;   // l.n - filters [16 - 55 - 1024]
    {
        j = index / M;  // out_h*out_w - one channel output size [169 - 173056]
        if (j < N)
        {
            int count = 0;
            k = 0;

//#ifdef NOT_USED
            // 32 thread X 64 bit = 2048 bit
            for (; k < (K - 2048); k += 2048) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t c_bit64;

                int64_t A_cur_index = (i*lda + k) / 8;
                //int64_t B_cur_index = (j*ldb + k) / 8;
                int64_t B_cur_index = (local_j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl(A_cur_index, t) + 8 * lane_id;
                    const int64_t B_i = __shfl(B_cur_index, t) + 8 * lane_id;

                    {
                        uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
                        //uint64_t b_bit64 = *((uint64_t *)(B + B_i));    // input
                        uint64_t b_bit64 = *((uint64_t *)(B_s + B_i));    // input
                        c_bit64 = xnor_int64(a_bit64, b_bit64);
                        int tmp_count = __popcll(c_bit64);

                        int sum_count = warpAllReduceSum(tmp_count);
                        if (lane_id == t) count += sum_count;
                    }
                }
            }
//#endif

//#ifdef NOT_USED
            // 32 thread X 32 bit = 1024 bit
            for (; k < (K - 1024); k += 1024) {   // l.size*l.size*l.c - one filter size [27 - 9216]

                int64_t A_cur_index = (i*lda + k) / 8;
                //int64_t B_cur_index = (j*ldb + k) / 8;
                int64_t B_cur_index = (local_j*ldb + k) / 8;
                if (i >= M) A_cur_index = 0;

                #pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    const int lane_id = threadIdx.x % WARP_SIZE;

                    const int64_t A_i = __shfl(A_cur_index, t) + 4 * lane_id;
                    const int64_t B_i = __shfl(B_cur_index, t) + 4 * lane_id;

                    {
                        uint32_t a_bit32 = *((uint32_t *)(A + A_i));    // weights
                        //uint32_t b_bit32 = *((uint32_t *)(B + B_i));    // input
                        uint32_t b_bit32 = *((uint32_t *)(B_s + B_i));    // input
                        uint32_t c_bit32 = xnor_int32(a_bit32, b_bit32);
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
                for (; k < K; k += 256) {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                    ulonglong4 a_bit256 = *((ulonglong4 *)(A + (i*lda + k) / 8));    // weights
                    //ulonglong4 b_bit256 = *((ulonglong4 *)(B + (j*ldb + k) / 8));    // input
                    ulonglong4 b_bit256 = *((ulonglong4 *)(B_s + (local_j*ldb + k) / 8));    // input
                    ulonglong4 c_bit256 = xnor_int256(a_bit256, b_bit256);

                    count += __popcll(c_bit256.w) + __popcll(c_bit256.x) +
                        __popcll(c_bit256.y) + __popcll(c_bit256.z);
                }
//#endif

#ifdef NOT_USED
                for (; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                    uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
                    //uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
                    uint64_t b_bit64 = *((uint64_t *)(B_s + (local_j*ldb + k) / 8));            // input
                    uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                    count += __popcll(c_bit64);
                }
#endif

                const int bit_step = 256;
                int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    // remove extra bits (from empty space for align only)

                C[i*ldc + j] = (2 * count - K) * mean_val + bias_val;
            }
        }
    }
}
*/

// GOOD
void gemm_nn_custom_bin_mean_transposed_gpu(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias)
{
    size_t size = M*N;
    const int num_blocks = size / BLOCK + 1;

    /*
    printf("\n gemm_bin size = %d, num_blocks = %d, M*K = %d KB, N*K = %d KB \n (w) M*K/num_blocks = %d KB, (i) N*K/num_blocks = %d KB \n",
        size, num_blocks, M*K / 1024, N*K / 1024, M*lda / num_blocks / 1024, N*ldb / num_blocks / 1024);
    printf(" M / 512 = %d, N / 512 = %d, M*lda / 512 = %d, N*ldb / 512 = %d \n", M / 512, N / 512, M*lda/512, N*ldb/512);
    */
    //printf(" shared_memory: (w) lda*BLOCK/N = %d, (i) ldb*BLOCK/M = %d, \t lda = %d \n\n", lda*BLOCK / N, ldb*BLOCK / M, lda);

    gemm_nn_custom_bin_mean_transposed_gpu_kernel<<<num_blocks, BLOCK, 0, get_cuda_stream() >>>(
        M, N, K,
        A, lda,
        B, ldb,
        C, ldc,
        mean_arr, bias);
}
// --------------------------------




// --------------------------------
// sequentially - B (input) in the shared_memory - BAD
// --------------------------------
__global__ void gemm_nn_custom_bin_mean_transposed_sequentially_gpu_kernel(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    //__shared__ float mean_shared[32];
    //__shared__ uint32_t B_s[8192];  // 32 KB // [ldb x N`] // max = 262 144 bits
    //__shared__ uint32_t B_s[4096];  // 16 KB // [ldb x N`] // max = 131 072 bits
    __shared__ uint8_t B_s[4096*4];  // 16 KB // [ldb x N`] // max = 131 072 bits


    const int K_items = WARP_SIZE;
    int start_j = blockIdx.x*blockDim.x / (K_items * M);

    {
        int end_j = (blockIdx.x*blockDim.x + blockDim.x) / (K_items * M) + 1;
        if (end_j > N) end_j = N;
        size_t shared_size = ldb * (end_j - start_j);

        if (shared_size != 0) {
            //if(threadIdx.x == 0) printf(" start_j = %d, end_j = %d, shared_size = %d \n", start_j, end_j, shared_size);

            int k;
            for (int k = threadIdx.x * 32; k < shared_size; k += blockDim.x * 32) {
                int x = start_j*ldb + k;
                if (x < (N*ldb)) *((uint32_t *)(B_s + k / 8)) = *((uint32_t *)(B + x / 8));
            }
        }
    }
    __syncthreads();

    int index = blockIdx.x*blockDim.x + threadIdx.x;

    {
        int i;  // l.n
        int j;  // out_h*out_w
        int k;  // l.size * l.size * l.c

        const int index2 = index / K_items;
        i = index2 % M; // max M
        j = index2 / M; // max N
                        //j = index2 % N; // max N
                        //i = index2 / N; // max M

                        //int j_cur = index / M;
                        //int local_j = j_cur - start_j;
        int local_j = j - start_j;

        //if (i <= 1 && j <= 1 ) printf(" k = %d, K = %d, K_items = %d, i = %d, j = %d, lda = %d, ldb = %d, ldc = %d \n",
        //    k, K, K_items, i, j, lda, ldb, ldc);
        {   // l.n - filters [16 - 55 - 1024]
            // further improvements: for (l.n == 1024) iterate several (j)


            if (j < N)
            { // out_h*out_w - one channel output size [169 - 173056]

                int count = 0;


                const int bit_step = 32;
                for (k = (threadIdx.x % WARP_SIZE) * bit_step; k < K; k += bit_step*WARP_SIZE)
                {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                    uint32_t a_bit32 = *((uint32_t *)(A + (i*lda + k) / 8));    // weights
                                                                                //uint32_t b_bit32 = *((uint32_t *)(B + (j*ldb + k) / 8));    // input
                    uint32_t b_bit32 = *((uint32_t *)(B_s + (local_j*ldb + k) / 8));    // input
                    uint32_t c_bit32 = xnor_int32(a_bit32, b_bit32);

                    count += __popc(c_bit32);
                }

                /*
                const int bit_step = 64;
                for (k = (threadIdx.x % WARP_SIZE) * bit_step; k < K; k += bit_step*WARP_SIZE)
                {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
                uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
                //uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));
                uint64_t b_bit64 = *((uint64_t *)(B_s + (local_j*ldb + k) / 8));    // input
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);
                count += __popcll(c_bit64);
                }
                */


                //atomicAdd(&C[i*ldc + j], (2 * count) * mean_val);

                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                    count += __shfl_down(count, offset);


                if (threadIdx.x % WARP_SIZE == 0) {
                    int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                    count = count - f1;
                    float mean_val = mean_arr[i];
                    C[i*ldc + j] = (2 * count - K) * mean_val;
                    //B_s[threadIdx.x / WARP_SIZE] = (2 * count - K) * mean_val;
                }
            }
        }
    }
}

// sequentially - BAD
void gemm_nn_custom_bin_mean_transposed_sequentially_gpu(int M, int N, int K,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    //size_t size = M*N;
    size_t size = M*N * 32;

    const int num_blocks = size / BLOCK + 1;

    //printf(" K = %d \n", K);

    /*
    printf("\n gemm_bin size = %d, num_blocks = %d, M*K = %d KB, N*K = %d KB \n (w) M*K/num_blocks = %d KB, (i) N*K/num_blocks = %d KB \n",
    size, num_blocks, M*K / 1024, N*K / 1024, M*lda / num_blocks / 1024, N*ldb / num_blocks / 1024);
    printf(" M / 512 = %d, N / 512 = %d, M*lda / 512 = %d, N*ldb / 512 = %d \n", M / 512, N / 512, M*lda/512, N*ldb/512);
    */
    //printf(" shared_memory: (w) lda*BLOCK/N = %d, (i) ldb*BLOCK/M = %d, \t lda = %d \n\n", lda*BLOCK / N, ldb*BLOCK / M, lda);

    gemm_nn_custom_bin_mean_transposed_sequentially_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(
        M, N, K,
        A, lda,
        B, ldb,
        C, ldc,
        mean_arr);
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
    size_t array_size = in_w*in_h*n;    // width X height X filters
    const int num_blocks = array_size / BLOCK + 1;
    //printf("\n array_size = %d, num_blocks = %d, w = %d, h = %d, n = %d, c = %d, pad = %d \n", array_size, num_blocks, in_w, in_h, n, in_c, pad);

    convolve_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (input, weights, output, in_w, in_h, in_c, n, size, pad);
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

        for (int fil = min_fil; fil <= max_fil; fil++) {
            for (int s = threadIdx.x; s < weights_size; s += blockDim.x) {
                //weights_shared[s + (fil - min_fil)*new_lda / 8] = ((uint8_t *)weights)[fil*new_lda / 8 + s];
                weights_shared[s + (fil - min_fil)*new_lda/32] = ((uint32_t *)weights)[fil*new_lda / 32 + s];
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
    size_t array_size = in_w*in_h*n;    // width X height X filters
    const int num_blocks = array_size / BLOCK + 1;
    //printf("\n array_size = %d, num_blocks = %d, w = %d, h = %d, n = %d, c = %d, pad = %d \n", array_size, num_blocks, in_w, in_h, n, in_c, pad);

    convolve_bin_gpu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (input, weights, output, in_w, in_h, in_c, n, size, pad, new_lda, mean_arr_gpu);
}

// --------------------------------

