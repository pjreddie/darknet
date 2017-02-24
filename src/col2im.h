#ifndef COL2IM_H
#define COL2IM_H

#ifdef __cplusplus
extern "C" {
#endif

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU
void col2im_ongpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);
#endif

#ifdef __cplusplus
}
#endif

#endif
