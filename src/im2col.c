#include "mini_blas.h"
#include <stdio.h>
inline float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}


#ifdef GPU

#include "opencl.h"
#include <math.h>

cl_kernel get_im2col_pad_kernel()
{
    static int init = 0;
    static cl_kernel im2col_kernel;
    if(!init){
        im2col_kernel = get_kernel("src/im2col.cl", "im2col_pad", 0);
        init = 1;
    }
    return im2col_kernel;
}

cl_kernel get_im2col_nopad_kernel()
{
    static int init = 0;
    static cl_kernel im2col_kernel;
    if(!init){
        im2col_kernel = get_kernel("src/im2col.cl", "im2col_nopad", 0);
        init = 1;
    }
    return im2col_kernel;
}


void im2col_ongpu(cl_mem data_im, int offset,
        int channels,  int height,  int width,
        int ksize,  int stride,  int pad, cl_mem data_col)
{
    cl_setup();

    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    cl_kernel kernel = get_im2col_nopad_kernel();

    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        kernel = get_im2col_pad_kernel();
    }

    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(data_im), (void*) &data_im);
    cl.error = clSetKernelArg(kernel, i++, sizeof(offset), (void*) &offset);
    cl.error = clSetKernelArg(kernel, i++, sizeof(channels), (void*) &channels);
    cl.error = clSetKernelArg(kernel, i++, sizeof(height), (void*) &height);
    cl.error = clSetKernelArg(kernel, i++, sizeof(width), (void*) &width);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ksize), (void*) &ksize);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(data_col), (void*) &data_col);
    check_error(cl);

    size_t global_size = channels_col*height_col*width_col;

    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0,
            &global_size, 0, 0, 0, 0);
    check_error(cl);
}

/*
   void im2col_gpu(float *data_im,
   int channels,  int height,  int width,
   int ksize,  int stride,  int pad, float *data_col) 
   {
   cl_setup();
   cl_context context = cl.context;
   cl_command_queue queue = cl.queue;

   size_t size = sizeof(float)*(channels*height*width*batch);
   cl_mem im_gpu = clCreateBuffer(context,
   CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
   size, data_im, &cl.error);
   check_error(cl);

   int height_col = (height - ksize) / stride + 1;
   int width_col = (width - ksize) / stride + 1;
   int channels_col = channels * ksize * ksize;

   size = sizeof(float)*(height_col*width_col*channels_col*batch);
   cl_mem col_gpu = clCreateBuffer(context,
   CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
   size, data_col, &cl.error);
   check_error(cl);

   im2col_ongpu(im_gpu, batch, channels, height, width,
   ksize, stride, pad, col_gpu);

   clEnqueueReadBuffer(queue, col_gpu, CL_TRUE, 0, size, data_col, 0, 0, 0);
   check_error(cl);

   clReleaseMemObject(col_gpu);
   clReleaseMemObject(im_gpu);
   }
 */

#endif
