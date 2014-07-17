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
void im2col_ongpu(cl_mem data_im, const int batch,
        const int channels, const int height, const int width,
        const int ksize, const int stride, cl_mem data_col);

void im2col_gpu(float *data_im,
    const int batch, const int channels, const int height, const int width,
    const int ksize, const int stride, float *data_col);

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        cl_mem A_gpu, int lda, 
        cl_mem B_gpu, int ldb,
        float BETA,
        cl_mem C_gpu, int ldc);
#endif

void im2col_cpu(float* data_im,
    const int channels, const int height, const int width,
    const int ksize, const int stride, int pad, float* data_col);

void col2im_cpu(float* data_col,
        const int channels, const int height, const int width,
        const int ksize, const int stride, int pad, float* data_im);
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
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void test_gpu_blas();
