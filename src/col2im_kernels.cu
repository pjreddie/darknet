#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include "col2im.h"
#include "dark_cuda.h"

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void col2im_gpu_kernel(const int n, const float* data_col,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_im) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset =
            (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void col2im_ongpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK, 0, get_cuda_stream() >>>(
                num_kernels, data_col, height, width, ksize, pad,
                stride, height_col,
                width_col, data_im);

    CHECK_CUDA(cudaPeekAtLastError());
}
// -----------------------------------------

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
__global__ void col2im_gpu_kernel_ext(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_im) {
    CUDA_KERNEL_LOOP(index, n) {
        float val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
            (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                        height_col + h_col) * width_col + w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = val;
    }
}

void col2im_gpu_ext(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im)
{
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
        stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
        stride_w + 1;
    int num_kernels = channels * height * width;
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
    col2im_gpu_kernel_ext<<<CAFFE_GET_BLOCKS(num_kernels),
        CAFFE_CUDA_NUM_THREADS >>>(
            num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            height_col, width_col, data_im);

    CHECK_CUDA(cudaPeekAtLastError());
}
