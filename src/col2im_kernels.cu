extern "C" {
#include "col2im.h"
#include "cuda.h"
}

__global__ void col2im_kernel(float *data_col, int offset,
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
    data_im[index+offset] = val;
}


extern "C" void col2im_ongpu(float *data_col, int offset,
        int channels,  int height,  int width,
        int ksize,  int stride,  int pad, float *data_im)
{

    size_t n = channels*height*width;

    col2im_kernel<<<cuda_gridsize(n), BLOCK>>>(data_col, offset, channels, height, width, ksize, stride, pad, data_im);
    check_error(cudaPeekAtLastError());
}
