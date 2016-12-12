#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

#ifdef __cplusplus
extern "C" {
#endif
void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);
#ifdef __cplusplus
}
#endif

#endif
#endif
