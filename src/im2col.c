#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
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
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

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


// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
    return (unsigned)(a) < (unsigned)(b);
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void im2col_cpu_ext(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col)
{
    const int output_h = (height + 2 * pad_h -
        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    int channel, kernel_row, kernel_col, output_rows, output_col;
    for (channel = channels; channel--; data_im += channel_size) {
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (output_col = output_w; output_col; output_col--) {
                            *(data_col++) = 0;
                        }
                    }
                    else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            }
                            else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}
