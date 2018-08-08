#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gemm_bin(int M, int N, int K, float ALPHA,
        char  *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}


//--------------------------------------------
// XNOR bitwise GEMM for binary neural network
//--------------------------------------------

#include <stdint.h>

static inline unsigned char xnor(unsigned char a, unsigned char b) {
    //return a == b;
    return !(a^b);
}

// INT-32
static inline uint32_t get_bit_int32(uint32_t const*const src, size_t index) {
    size_t src_i = index / 32;
    int src_shift = index % 32;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    return val;
}

static inline uint32_t xnor_int32(uint32_t a, uint32_t b) {
    return ~(a^b);
}

static inline uint64_t xnor_int64(uint64_t a, uint64_t b) {
    return ~(a^b);
}


static inline uint32_t fill_bit_int32(char src) {
    if (src == 0) return 0x00000000;
    else return  0xFFFFFFFF;
}

static inline uint64_t fill_bit_int64(char src) {
    if (src == 0) return 0x0000000000000000;
    else return  0xFFFFFFFFFFFFFFFF;
}

void binary_int32_printf(uint32_t src) {
    int i;
    for (i = 0; i < 32; ++i) {
        if (src & 1) printf("1");
        else printf("0");
        src = src >> 1;
    }
    printf("\n");
}

void binary_int64_printf(uint64_t src) {
    int i;
    for (i = 0; i < 64; ++i) {
        if (src & 1) printf("1");
        else printf("0");
        src = src >> 1;
    }
    printf("\n");
}

/*
void gemm_nn_custom_bin_mean(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int *count_arr = calloc(M*N, sizeof(int));

    int i, j, k;
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        for (k = 0; k < K; ++k) {   // l.size*l.size*l.c - one filter size [27 - 9216]
            char a_bit = get_bit(A, i*lda + k);

            for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
                char b_bit = get_bit(B, k*ldb + j);
                count_arr[i*ldc + j] += xnor(a_bit, b_bit);
            }
        }
    }

    for (i = 0; i < M; ++i) {
        float mean_val = mean_arr[i];
        for (j = 0; j < N; ++j) {
            C[i*ldc + j] = (2 * count_arr[i*ldc + j] - K) * mean_val;
        }
    }
    free(count_arr);
}
*/

/*
void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int *count_arr = calloc(M*N, sizeof(int));

    int i, j, k;
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            for (k = 0; k < K; ++k) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                char a_bit = get_bit(A, i*lda + k);
                char b_bit = get_bit(B, j*ldb + k);
                count_arr[i*ldc + j] += xnor(a_bit, b_bit);
            }
        }
    }

    for (i = 0; i < M; ++i) {
        float mean_val = mean_arr[i];
        for (j = 0; j < N; ++j) {
            C[i*ldc + j] = (2 * count_arr[i*ldc + j] - K) * mean_val;
        }
    }
    free(count_arr);
}
*/

/*
void gemm_nn_custom_bin_mean(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int *count_arr = calloc(M*N, sizeof(int));

    int i, j, k, h;

#pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        for (k = 0; k < K; ++k) {   // l.size*l.size*l.c - one filter size [27 - 9216]
            const char a_bit = get_bit(A, i*lda + k);
            uint64_t a_bit64 = fill_bit_int64(a_bit);
            int  k_ldb = k*ldb;

            for (j = 0; j < N; j += 64) { // out_h*out_w - one channel output size [169 - 173056]
                if ((N - j > 64) && (k_ldb % 8 == 0)) {
                    uint64_t b_bit64 = *((uint64_t *)(B + (k_ldb + j) / 8));
                    uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);
                    //printf("\n %d \n",__builtin_popcountll(c_bit64)); // gcc
                    printf("\n %d \n", __popcnt64(c_bit64));    // msvs

                    int h;
                    for (h = 0; h < 64; ++h)
                        if ((c_bit64 >> h) & 1) count_arr[i*ldc + j + h] += 1;

                    //binary_int64_printf(a_bit64);
                    //binary_int64_printf(b_bit64);
                    //binary_int64_printf(c_bit64);
                }
                else {
                    for (; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
                        char b_bit = get_bit(B, k_ldb + j);
                        if (xnor(a_bit, b_bit)) count_arr[i*ldc + j] += 1;
                    }
                }

            }
        }
    }

    if (mean_arr) {
        //int K_2 = K / 2;
        for (i = 0; i < M; ++i) {
            float mean_val = mean_arr[i];
            //float mean_val2 = 2 * mean_val;
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] = (2 * count_arr[i*ldc + j] - K) * mean_val;
                //C[i*ldc + j] = (count_arr[i*ldc + j] - K_2) *mean_val2;
            }
        }
    }
    else {
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] = count_arr[i*ldc + j] - K / 2;
            }
        }
    }

    free(count_arr);

    //getchar();
}
*/


/*
void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i, j, k, h;

#pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        float mean_val = mean_arr[i];

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));
                uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

#ifdef WIN32
                int tmp_count = __popcnt64(c_bit64);
#else
                int tmp_count = __builtin_popcountll(c_bit64);
#endif

                if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                count += tmp_count;
                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}
*/

//----------------------------


#if (defined(__AVX__) && defined(__x86_64__)) || defined(_WIN64)

#define OSXSAVEFlag (1UL<<27)
#define AVXFlag     ((1UL<<28)|OSXSAVEFlag)
#define FMAFlag     ((1UL<<12)|AVXFlag|OSXSAVEFlag)
#define CLMULFlag   ((1UL<< 1)|AVXFlag|OSXSAVEFlag)
#define VAESFlag    ((1UL<<25)|AVXFlag|OSXSAVEFlag)

#ifdef _WIN64
#include <intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#else    // Linux GCC/Clang
#include <x86intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <cpuid.h>

void asm_cpuid(uint32_t* abcd, uint32_t eax)
{
    uint32_t ebx = 0, edx = 0, ecx = 0;

    // EBX is saved to EDI and later restored
    __asm__("movl %%ebx, %%edi;"
        "cpuid;"
        "xchgl %%ebx, %%edi;"
        : "=D"(ebx),
        "+a"(eax), "+c"(ecx), "=d"(edx));

    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
}

#endif

int simd_detect_x86(unsigned int idFeature)
{
    uint32_t regs[4];    // EAX, EBX, ECX, EDX;
#ifdef _WIN32
    __cpuid(regs, 0);
    if (regs[0] > 1U) __cpuid(regs, 1);
#else
    __get_cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3]);
    if(regs[0] > 1U) __get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if ((regs[2] & idFeature) != idFeature)
        return 0;
    return 1;
}

int is_fma_avx() {
    static int result = -1;
    if (result == -1) {
        result = simd_detect_x86(AVXFlag);
        if (result == 1) printf(" Used AVX \n");
        else printf(" Not used AVX \n");
    }
    return result;
}

// https://software.intel.com/sites/landingpage/IntrinsicsGuide
void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    if (is_fma_avx() == 1) {    // AVX
        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                float A_PART = ALPHA*A[i*lda + k];
                __m256 a256, b256, c256, result256;    // AVX
                a256 = _mm256_set1_ps(A_PART);
                for (j = 0; j < N - 8; j += 8) {
                    b256 = _mm256_loadu_ps(&B[k*ldb + j]);
                    c256 = _mm256_loadu_ps(&C[i*ldc + j]);
                    // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                    //result256 = _mm256_fmadd_ps(a256, b256, c256);
                    result256 = _mm256_mul_ps(a256, b256);
                    result256 = _mm256_add_ps(result256, c256);
                    _mm256_storeu_ps(&C[i*ldc + j], result256);
                }

                int prev_end = (N % 8 == 0) ? (N - 8) : (N / 8) * 8;
                for (j = prev_end; j < N; ++j)
                    C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
    else {
        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                register float A_PART = ALPHA*A[i*lda + k];
                for (j = 0; j < N; ++j) {
                    C[i*ldc + j] += A_PART*B[k*ldb + j];
                }
                /* // SSE
                __m128 a128, b128, c128, result128;    // SSE
                a128 = _mm_set1_ps(A_PART);
                for (j = 0; j < N - 4; j += 4) {
                b128 = _mm_loadu_ps(&B[k*ldb + j]);
                c128 = _mm_loadu_ps(&C[i*ldc + j]);
                //result128 = _mm_fmadd_ps(a128, b128, c128);
                result128 = _mm_mul_ps(a128, b128);
                result128 = _mm_add_ps(result128, c128);
                _mm_storeu_ps(&C[i*ldc + j], result128);
                }

                int prev_end = (N % 4 == 0) ? (N - 4) : (N / 4) * 4;
                for (j = prev_end; j < N; ++j){
                C[i*ldc + j] += A_PART*B[k*ldb + j];
                }
                */
            }
        }
    }
}


// http://graphics.stanford.edu/~seander/bithacks.html
// https://stackoverflow.com/questions/17354971/fast-counting-the-number-of-set-bits-in-m128i-register


static inline int popcnt128(__m128i n) {
    const __m128i n_hi = _mm_unpackhi_epi64(n, n);
#ifdef _MSC_VER
    return __popcnt64(_mm_cvtsi128_si64(n)) + __popcnt64(_mm_cvtsi128_si64(n_hi));
#else
    return __popcntq(_mm_cvtsi128_si64(n)) + __popcntq(_mm_cvtsi128_si64(n_hi));
#endif
}

static inline int popcnt256(__m256i n) {
    return popcnt128(_mm256_extractf128_si256(n, 0)) + popcnt128(_mm256_extractf128_si256(n, 1));
}

static inline __m256i count256(__m256i v) {
    __m256i lookup =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
            2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
            1, 2, 2, 3, 2, 3, 3, 4);

    __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i total = _mm256_add_epi8(popcnt1, popcnt2);

    return _mm256_sad_epu8(total, _mm256_setzero_si256());
}
static inline int popcnt256_custom(__m256i n) {
    return _mm_popcnt_u64(n.m256i_i64[0]) +
        _mm_popcnt_u64(n.m256i_i64[1]) +
        _mm_popcnt_u64(n.m256i_i64[2]) +
        _mm_popcnt_u64(n.m256i_i64[3]);
}

static inline void CSA(__m256i * h, __m256i * l, __m256i a, __m256i b, __m256i c)
{
    __m256i u = _mm256_xor_si256(a, b);
    *h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    *l = _mm256_xor_si256(u, c);
}

static inline __m256i xnor256(__m256i a_bit256, __m256i b_bit256) {
    __m256i all_1 = _mm256_set1_epi8(255);
    __m256i xor256 = _mm256_xor_si256(a_bit256, b_bit256);
    __m256i c_bit256 = _mm256_andnot_si256(xor256, all_1);

    return c_bit256;

}

// 2 x faster than popcnt: https://arxiv.org/pdf/1611.07612.pdf
// step = 16*256/8 = 512 bytes = 4096 bit (ldb, lda, bit_step, align - all should be aligned by 4096 bit)
static inline uint64_t avx_hs_custom(__m256i * A, __m256i * B, uint64_t size) {
    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256();
    __m256i twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    for (uint64_t i = 0; i < size; i += 16) {
        //CSA(&twosA, &ones, ones, d[i], d[i + 1]);
        CSA(&twosA, &ones, ones, xnor256(A[i], B[i]), xnor256(A[i + 1], B[i + 1]));
        CSA(&twosB, &ones, ones, xnor256(A[i + 2], B[i + 2]), xnor256(A[i + 3], B[i + 3]));
        CSA(&foursA, &twos, twos, twosA, twosB);
        CSA(&twosA, &ones, ones, xnor256(A[i + 4], B[i + 4]), xnor256(A[i + 5], B[i + 5]));
        CSA(&twosB, &ones, ones, xnor256(A[i + 6], B[i + 6]), xnor256(A[i + 7], B[i + 7]));
        CSA(&foursB, &twos, twos, twosA, twosB);
        CSA(&eightsA, &fours, fours, foursA, foursB);
        CSA(&twosA, &ones, ones, xnor256(A[i + 8], B[i + 8]), xnor256(A[i + 9], B[i + 9]));
        CSA(&twosB, &ones, ones, xnor256(A[i + 10], B[i + 10]), xnor256(A[i + 11], B[i + 11]));
        CSA(&foursA, &twos, twos, twosA, twosB);
        CSA(&twosA, &ones, ones, xnor256(A[i + 12], B[i + 12]), xnor256(A[i + 13], B[i + 13]));
        CSA(&twosB, &ones, ones, xnor256(A[i + 14], B[i + 14]), xnor256(A[i + 15], B[i + 15]));
        CSA(&foursB, &twos, twos, twosA, twosB);
        CSA(&eightsB, &fours, fours, foursA, foursB);
        CSA(&sixteens, &eights, eights, eightsA, eightsB);

        total = _mm256_add_epi64(total, count256(sixteens));
    }
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total,
        _mm256_slli_epi64(count256(eights), 3));
    total = _mm256_add_epi64(total,
        _mm256_slli_epi64(count256(fours), 2));
    total = _mm256_add_epi64(total,
        _mm256_slli_epi64(count256(twos), 1));
    total = _mm256_add_epi64(total, count256(ones));

    return total.m256i_i64[0] +
            total.m256i_i64[1] +
            total.m256i_i64[2] +
            total.m256i_i64[3];

    //return _mm256_extract_epi64(total, 0)
    //    + _mm256_extract_epi64(total, 1)
    //    + _mm256_extract_epi64(total, 2)
    //    + _mm256_extract_epi64(total, 3);
}

void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    __m256i all_1 = _mm256_set1_epi8(255);
    int i, j, k;

    //printf("\n M = %d, N = %d, K = %d, ldb = %d, M*ldb/8 = %d, N*ldb/8= %d \n", M, N, K, ldb, M*ldb/8, N*ldb/8);
    //if (K > 4096)  printf("!!!avx_hs!!! \n\n");

    #pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        float mean_val = mean_arr[i];

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;
            const int bit_step = 256;


            int hs_count = 0;
            if (K > 4096) {
                hs_count = avx_hs_custom(A + (i*lda) / 8, B + (j*ldb) / 8, K / 256);

                int local_bit_step = 4096;

                int f1 = (K % local_bit_step == 0) ? 0 : (local_bit_step - (K % local_bit_step));
                hs_count = hs_count - f1;    // remove extra bits
                count = hs_count;
            }
            else {
                for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]

                    //__m128i a_bit128 = _mm_loadu_si128((__m128i *)(A + (i*lda + k) / 8));
                    //__m128i b_bit128 = _mm_loadu_si128((__m128i *)(B + (j*ldb + k) / 8));
                    //__m128i xor128 = _mm_xor_si128(a_bit128, b_bit128);
                    //__m128i c_bit128 = _mm_andnot_si128(xor128, all_1);
                    //int tmp_count = popcnt128(c_bit128);

                    __m256i a_bit256 = _mm256_loadu_si256((__m256i *)(A + (i*lda + k) / 8));
                    __m256i b_bit256 = _mm256_loadu_si256((__m256i *)(B + (j*ldb + k) / 8));
                    __m256i xor256 = _mm256_xor_si256(a_bit256, b_bit256);
                    __m256i c_bit256 = _mm256_andnot_si256(xor256, all_1); //we can do NOT for wegihts once and do not do this NOT
                    int tmp_count = popcnt256(c_bit256);
                    //int tmp_count = popcnt256_custom(c_bit256);
                    count += tmp_count;

                    //binary_int64_printf(c_bit64);
                    //printf(", count = %d \n\n", tmp_count);
                }

                int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    // remove extra bits
           }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}


void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);

    for (i = 0; i < size; i+=8)
    {
        __m256i src256 = _mm256_loadu_si256((__m256i *)(&src[i]));
        __m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats

        uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
        mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

        dst[i / 8] = mask;
    }
}

#else

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i, j, k, h;

#pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        float mean_val = mean_arr[i];

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));
                uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

#ifdef WIN32
                int tmp_count = __popcnt64(c_bit64);
#else
                int tmp_count = __builtin_popcountll(c_bit64);
#endif

                if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                count += tmp_count;
                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}

void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    char *byte_arr = calloc(size, sizeof(char));
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) byte_arr[i] = 1;
    }

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for (i = 0; i < size; i += 8) {
        char dst_tmp = 0;
        dst_tmp |= byte_arr[i + 0] << 0;
        dst_tmp |= byte_arr[i + 1] << 1;
        dst_tmp |= byte_arr[i + 2] << 2;
        dst_tmp |= byte_arr[i + 3] << 3;
        dst_tmp |= byte_arr[i + 4] << 4;
        dst_tmp |= byte_arr[i + 5] << 5;
        dst_tmp |= byte_arr[i + 6] << 6;
        dst_tmp |= byte_arr[i + 7] << 7;
        dst[i / 8] = dst_tmp;
    }
    free(byte_arr);
}
#endif    // __x86_64

void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    if (BETA != 1){
        int i, j;
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                C[i*ldc + j] *= BETA;
            }
        }
    }

    int t;
    #pragma omp parallel for
    for (t = 0; t < M; ++t) {
        if (!TA && !TB)
            gemm_nn(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
        else if (TA && !TB)
            gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
        else if (!TA && TB)
            gemm_nt(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
        else
            gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
    }
}

#ifdef GPU

#include <math.h>

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t stream_status = cublasSetStream(handle, get_cuda_stream());
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    float *A_gpu = cuda_make_array(A, (TA ? lda*K:lda*M));
    float *B_gpu = cuda_make_array(B, (TB ? ldb*N : ldb*K));
    float *C_gpu = cuda_make_array(C, ldc*M);

    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_ongpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_ongpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75);

       test_gpu_accuracy(0,0,17,10,10);
       test_gpu_accuracy(1,0,17,10,10);
       test_gpu_accuracy(0,1,17,10,10);
       test_gpu_accuracy(1,1,17,10,10);

       test_gpu_accuracy(0,0,1000,10,100);
       test_gpu_accuracy(1,0,1000,10,100);
       test_gpu_accuracy(0,1,1000,10,100);
       test_gpu_accuracy(1,1,1000,10,100);

       test_gpu_accuracy(0,0,10,10,10);

       time_ongpu(0,0,64,2916,363);
       time_ongpu(0,0,64,2916,363);
       time_ongpu(0,0,64,2916,363);
       time_ongpu(0,0,192,729,1600);
       time_ongpu(0,0,384,196,1728);
       time_ongpu(0,0,256,196,3456);
       time_ongpu(0,0,256,196,2304);
       time_ongpu(0,0,128,4096,12544);
       time_ongpu(0,0,128,4096,4096);
     */
    time_ongpu(0,0,64,75,12544);
    time_ongpu(0,0,64,75,12544);
    time_ongpu(0,0,64,75,12544);
    time_ongpu(0,0,64,576,12544);
    time_ongpu(0,0,256,2304,784);
    time_ongpu(1,1,2304,256,784);
    time_ongpu(0,0,512,4608,196);
    time_ongpu(1,1,4608,512,196);

    return 0;
}
#endif

