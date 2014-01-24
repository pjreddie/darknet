void gemm(int TA, int TB, int M, int N, int K, double ALPHA, 
                    double *A, int lda, 
                    double *B, int ldb,
                    double BETA,
                    double *C, int ldc);
void im2row(double *image, int h, int w, int c, int size, int stride, double *matrix);
void im2col(double *image, int h, int w, int c, int size, int stride, double *matrix);
void im2col_cpu(double* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride,
        double* data_col);
