#ifndef GEMM_H
#define GEMM_H

static inline void set_bit(unsigned char *const dst, size_t index) {
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
}

static inline unsigned char get_bit(unsigned char const*const src, size_t index) {
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    return val;
}

void float_to_bit(float *src, unsigned char *dst, size_t size);

void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr);


//void gemm_nn_custom_bin_mean(int M, int N, int K, float ALPHA_UNUSED,
    //unsigned char *A, int lda,
    //unsigned char *B, int ldb,
    //float *C, int ldc, float *mean_arr)



void gemm_bin(int M, int N, int K, float ALPHA,
        char  *A, int lda,
        float *B, int ldb,
        float *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif
#endif
