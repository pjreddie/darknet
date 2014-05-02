void pm(int M, int N, float *A);
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void im2col_gpu(float* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_col);
void im2col_cpu(float* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_col);
void col2im_cpu(float* data_col, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_im);
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
