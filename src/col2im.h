#ifndef COL2IM_H
#define COL2IM_H

#ifdef __cplusplus
extern "C" {
#endif
void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

void col2im_cpu_ext(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im);

#ifdef GPU
void col2im_ongpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);


void col2im_gpu_ext(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
#endif
#ifdef __cplusplus
}
#endif
#endif
