#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
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
    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}
/*
   __global__ void im2col_pad_kernel(float *im,
   int channels,  int height,  int width,
   int ksize,  int stride, float *data_col)
   {
   int c,h,w;
   int height_col = 1 + (height-1) / stride;
   int width_col = 1 + (width-1) / stride;
   int channels_col = channels * ksize * ksize;

   int pad = ksize/2;

   int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
   int col_size = height_col*width_col*channels_col;
   if (id >= col_size) return;

   int col_index = id;
   w = id % width_col;
   id /= width_col;
   h = id % height_col;
   id /= height_col;
   c = id % channels_col;
   id /= channels_col;

   int w_offset = c % ksize;
   int h_offset = (c / ksize) % ksize;
   int im_channel = c / ksize / ksize;
   int im_row = h_offset + h * stride - pad;
   int im_col = w_offset + w * stride - pad;

   int im_index = im_col + width*(im_row + height*im_channel);
   float val = (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width) ? 0 : im[im_index];

   data_col[col_index] = val;
   }

   __global__ void im2col_nopad_kernel(float *im,
   int channels,  int height,  int width,
   int ksize,  int stride, float *data_col)
   {
   int c,h,w;
   int height_col = (height - ksize) / stride + 1;
   int width_col = (width - ksize) / stride + 1;
   int channels_col = channels * ksize * ksize;

   int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
   int col_size = height_col*width_col*channels_col;
   if (id >= col_size) return;

   int col_index = id;
   w = id % width_col;
   id /= width_col;
   h = id % height_col;
   id /= height_col;
   c = id % channels_col;
   id /= channels_col;

   int w_offset = c % ksize;
   int h_offset = (c / ksize) % ksize;
   int im_channel = c / ksize / ksize;
   int im_row = h_offset + h * stride;
   int im_col = w_offset + w * stride;

   int im_index = im_col + width*(im_row + height*im_channel);
   float val = (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width) ? 0 : im[im_index];

   data_col[col_index] = val;
   }

   extern "C" void im2col_ongpu(float *im,
   int channels,  int height,  int width,
int ksize,  int stride,  int pad, float *data_col)
{

    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
    }

    size_t n = channels_col*height_col*width_col;

    if(pad)im2col_pad_kernel<<<cuda_gridsize(n),BLOCK>>>(im,  channels, height, width, ksize, stride, data_col);
    else im2col_nopad_kernel<<<cuda_gridsize(n),BLOCK>>>(im,  channels, height, width, ksize, stride, data_col);
    check_error(cudaPeekAtLastError());
}
*/
