#include "opencl.h"

void pm(int M, int N, float *A);
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);

#ifdef GPU
void axpy_ongpu(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY);
void copy_ongpu(int N, cl_mem X, int INCX, cl_mem Y, int INCY);
void scal_ongpu(int N, float ALPHA, cl_mem X, int INCX);
void im2col_ongpu(cl_mem data_im, int batch,
         int channels, int height, int width,
         int ksize, int stride, int pad, cl_mem data_col);

void col2im_gpu(float *data_col,  int batch,
         int channels,  int height,  int width,
         int ksize,  int stride,  int pad, float *data_im);
void col2im_ongpu(cl_mem data_col, int batch,
        int channels, int height, int width,
        int ksize, int stride, int pad, cl_mem data_im);

void im2col_gpu(float *data_im, int batch,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col);

void gemm_ongpu_offset(int TA, int TB, int M, int N, int K, float ALPHA, 
        cl_mem A_gpu, int a_off, int lda, 
        cl_mem B_gpu, int b_off, int ldb,
        float BETA,
        cl_mem C_gpu, int c_off, int ldc);

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        cl_mem A_gpu, int lda, 
        cl_mem B_gpu, int ldb,
        float BETA,
        cl_mem C_gpu, int ldc);
#endif

void im2col_cpu(float* data_im, int batch,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col);

void col2im_cpu(float* data_col, int batch,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

void test_blas();

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void test_gpu_blas();
