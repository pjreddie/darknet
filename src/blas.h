#ifndef BLAS_H
#define BLAS_H
#include <stdlib.h>
#include "darknet.h"

#ifdef GPU
#include "dark_cuda.h"
#include "tree.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void const_cpu(int N, float ALPHA, float *X, int INCX);
void constrain_ongpu(int N, float ALPHA, float * X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);

void softmax(float *input, int n, float temp, float *output, int stride);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void constrain_cpu(int size, float ALPHA, float *X);
void fix_nan_and_inf_cpu(float *input, size_t size);

#ifdef GPU

void axpy_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void axpy_ongpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void simple_copy_ongpu(int size, float *src, float *dst);
void copy_ongpu(int N, float * X, int INCX, float * Y, int INCY);
void copy_ongpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void scal_ongpu(int N, float ALPHA, float * X, int INCX);
void scal_add_ongpu(int N, float ALPHA, float BETA, float * X, int INCX);
void supp_ongpu(int N, float ALPHA, float * X, int INCX);
void mask_gpu_new_api(int N, float * X, float mask_num, float * mask, float val);
void mask_ongpu(int N, float * X, float mask_num, float * mask);
void const_ongpu(int N, float ALPHA, float *X, int INCX);
void pow_ongpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_ongpu(int N, float *X, int INCX, float *Y, int INCY);
void fill_ongpu(int N, float ALPHA, float * X, int INCX);

void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);

void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);
void input_shortcut_gpu(float *in, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);

void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
void mult_add_into_gpu(int num, float *a, float *b, float *c);

void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void softmax_gpu_new_api(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void softmax_gpu(float *input, int n, int offset, int groups, float temp, float *output);
void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);

void flatten_ongpu(float *x, int spatial, int layers, int batch, int forward, float *out);

void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

void softmax_tree_gpu(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier);

void fix_nan_and_inf(float *input, size_t size);
int is_nan_or_inf(float *input, size_t size);

void add_3_arrays_activate(float *a1, float *a2, float *a3, size_t size, ACTIVATION a, float *dst);
void sum_of_mults(float *a1, float *a2, float *b1, float *b2, size_t size, float *dst);
void activate_and_mult(float *a1, float *a2, size_t size, ACTIVATION a, float *dst);

void scale_channels_gpu(float *in_w_h_c, int size, int channel_size, float *scales_c, float *out);
void backward_scale_channels_gpu(float *in_w_h_c_delta, int size, int channel_size,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta);


void backward_sam_gpu(float *in_w_h_c_delta, int size, int channel_size,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta);

void sam_gpu(float *in_w_h_c, int size, int channel_size, float *scales_c, float *out);

#endif
#ifdef __cplusplus
}
#endif
#endif
