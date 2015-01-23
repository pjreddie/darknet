extern "C" {
#include "im2col.h"
#include "cuda.h"
}

__global__ void im2col_pad_kernel(float *im,  int offset,
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

    int im_index = offset + im_col + width*(im_row + height*im_channel);
    float val = (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width) ? 0 : im[im_index];

    data_col[col_index] = val;
}

__global__ void im2col_nopad_kernel(float *im,  int offset,
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

    int im_index = offset + im_col + width*(im_row + height*im_channel);
    float val = (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width) ? 0 : im[im_index];

    data_col[col_index] = val;
}

extern "C" void im2col_ongpu(float *im, int offset,
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

    if(pad)im2col_pad_kernel<<<cuda_gridsize(n),BLOCK>>>(im,  offset, channels, height, width, ksize, stride, data_col);
    else im2col_nopad_kernel<<<cuda_gridsize(n),BLOCK>>>(im,  offset, channels, height, width, ksize, stride, data_col);
    check_error(cudaPeekAtLastError());
}
