extern "C" {
#include "col2im.h"
#include "cuda.h"
}

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
        data_im[index] = val;
    }
}

void col2im_ongpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, data_col, height, width, ksize, pad,
                stride, height_col,
                width_col, data_im);
}

/*
   __global__ void col2im_kernel(float *data_col,
   int channels, int height, int width,
   int ksize, int stride, int pad, float *data_im)
   {

   int height_col = (height - ksize) / stride + 1;
   int width_col = (width - ksize) / stride + 1;
   if (pad){
   height_col = 1 + (height-1) / stride;
   width_col = 1 + (width-1) / stride;
   pad = ksize/2;
   }

   int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
   if(id >= channels*height*width) return;

   int index = id;
   int w = id%width + pad;
   id /= width;
   int h = id%height + pad;
   id /= height;
   int c = id%channels;

   int w_start = (w-ksize+stride)/stride;
   int w_end = w/stride + 1;

   int h_start = (h-ksize+stride)/stride;
   int h_end = h/stride + 1;

// int rows = channels * ksize * ksize;
// int cols = height_col*width_col;
int col_offset = (c*ksize*ksize + h * ksize + w)*height_col*width_col;
int h_coeff = (1-stride*ksize*height_col)*width_col;
int w_coeff = 1-stride*height_col*width_col;
float val = 0;
int h_col, w_col;
for(h_col = h_start; h_col < h_end; ++h_col){
for(w_col = w_start; w_col < w_end; ++w_col){
int col_index = col_offset +h_col*h_coeff + w_col*w_coeff;
float part = (w_col < 0 || h_col < 0 || h_col >= height_col || w_col >= width_col) ? 0 : data_col[col_index];
val += part;
}
}
data_im[index] = val;
}


extern "C" void col2im_ongpu(float *data_col,
int channels,  int height,  int width,
int ksize,  int stride,  int pad, float *data_im)
{

size_t n = channels*height*width;

col2im_kernel<<<cuda_gridsize(n), BLOCK>>>(data_col, channels, height, width, ksize, stride, pad, data_im);
check_error(cudaPeekAtLastError());
}
 */
