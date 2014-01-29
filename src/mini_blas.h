void pm(int M, int N, float *A);
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);
void im2row(float *image, int h, int w, int c, int size, int stride, float *matrix);
void im2col(float *image, int h, int w, int c, int size, int stride, float *matrix);
void im2col_cpu(float* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_col);
void col2im_cpu(float* data_col, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_im);
void test_blas();
