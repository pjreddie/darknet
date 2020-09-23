#include "gemm.h"
#include "utils.h"
#include "im2col.h"
#include "dark_cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <intrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

#define TILE_M 4 // 4 ops
#define TILE_N 16 // AVX2 = 2 ops * 8 floats
#define TILE_K 16 // loop
#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

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
    float* m = (float*)xcalloc(rows * cols, sizeof(float));
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
    int *count_arr = xcalloc(M*N, sizeof(int));

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
    int *count_arr = xcalloc(M*N, sizeof(int));

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
    int *count_arr = xcalloc(M*N, sizeof(int));

    int i;

#pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        int j, k, h;
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
    int i;

#pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        int j, k, h;
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

// is not used
/*
void transpose_32x32_bits_my(uint32_t *A, uint32_t *B, int lda, int ldb)
{
    unsigned int x, y;
    for (y = 0; y < 32; ++y) {
        for (x = 0; x < 32; ++x) {
            if (A[y * lda] & ((uint32_t)1 << x)) B[x * ldb] |= (uint32_t)1 << y;
        }
    }
}
*/

#ifndef GPU
uint8_t reverse_8_bit(uint8_t a) {
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

uint32_t reverse_32_bit(uint32_t a)
{
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return (reverse_8_bit(a >> 24) << 0) |
        (reverse_8_bit(a >> 16) << 8) |
        (reverse_8_bit(a >> 8) << 16) |
        (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

void transpose32_optimized(uint32_t A[32]) {
    int j, k;
    unsigned m, t;

    //m = 0x0000FFFF;
    //for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
    //    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
    //        t = (A[k] ^ (A[k + j] >> j)) & m;
    //        A[k] = A[k] ^ t;
    //        A[k + j] = A[k + j] ^ (t << j);
    //    }
    //}

    j = 16;
    m = 0x0000FFFF;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 8;
    m = 0x00ff00ff;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 4;
    m = 0x0f0f0f0f;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 2;
    m = 0x33333333;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    j = 1;
    m = 0x55555555;
    for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

    // reverse Y
    for (j = 0; j < 16; ++j) {
        uint32_t tmp = A[j];
        A[j] = reverse_32_bit(A[31 - j]);
        A[31 - j] = reverse_32_bit(tmp);
    }
}

void transpose_32x32_bits_reversed_diagonale(uint32_t *A, uint32_t *B, int m, int n)
{
    unsigned A_tmp[32];
    int i;
    #pragma unroll
    for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
    transpose32_optimized(A_tmp);
    #pragma unroll
    for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];
}


void transpose_8x8_bits_my(unsigned char *A, unsigned char *B, int lda, int ldb)
{
    unsigned x, y;
    for (y = 0; y < 8; ++y) {
        for (x = 0; x < 8; ++x) {
            if (A[y * lda] & (1 << x)) B[x * ldb] |= 1 << y;
        }
    }
}

unsigned char reverse_byte_1(char a)
{
    return ((a & 0x1) << 7) | ((a & 0x2) << 5) |
        ((a & 0x4) << 3) | ((a & 0x8) << 1) |
        ((a & 0x10) >> 1) | ((a & 0x20) >> 3) |
        ((a & 0x40) >> 5) | ((a & 0x80) >> 7);
}

unsigned char reverse_byte(unsigned char a)
{
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

static unsigned char lookup[16] = {
    0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
    0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf, };

unsigned char reverse_byte_3(unsigned char n) {
    // Reverse the top and bottom nibble then swap them.
    return (lookup[n & 0b1111] << 4) | lookup[n >> 4];
}


void transpose8rS32_reversed_diagonale(unsigned char* A, unsigned char* B, int m, int n)
{
    unsigned x, y, t;

    x = y = 0;
    // Load the array and pack it into x and y.
    //x = (A[0] << 24) | (A[m] << 16) | (A[2 * m] << 8) | A[3 * m];
    //y = (A[4 * m] << 24) | (A[5 * m] << 16) | (A[6 * m] << 8) | A[7 * m];

    t = (x ^ (x >> 7)) & 0x00AA00AA;  x = x ^ t ^ (t << 7);
    t = (y ^ (y >> 7)) & 0x00AA00AA;  y = y ^ t ^ (t << 7);

    t = (x ^ (x >> 14)) & 0x0000CCCC;  x = x ^ t ^ (t << 14);
    t = (y ^ (y >> 14)) & 0x0000CCCC;  y = y ^ t ^ (t << 14);

    t = (x & 0xF0F0F0F0) | ((y >> 4) & 0x0F0F0F0F);
    y = ((x << 4) & 0xF0F0F0F0) | (y & 0x0F0F0F0F);
    x = t;

    B[7 * n] = reverse_byte(x >> 24);  B[6 * n] = reverse_byte(x >> 16);  B[5 * n] = reverse_byte(x >> 8);  B[4 * n] = reverse_byte(x);
    B[3 * n] = reverse_byte(y >> 24);  B[2 * n] = reverse_byte(y >> 16);  B[1 * n] = reverse_byte(y >> 8);  B[0 * n] = reverse_byte(y);
}

/*
// transpose by 8-bit
void transpose_bin(char *A, char *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    //printf("\n n = %d, ldb = %d \t\t m = %d, lda = %d \n", n, ldb, m, lda);
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += 8) {
        int j;
        for (j = 0; j < m; j += 8) {
            int a_index = i*lda + j;
            int b_index = j*ldb + i;
            //transpose_8x8_bits_my(&A[a_index/8], &B[b_index/8], lda/8, ldb/8);
            transpose8rS32_reversed_diagonale(&A[a_index / 8], &B[b_index / 8], lda / 8, ldb / 8);
        }
        for (; j < m; ++j) {
            if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
        }
    }
}
*/

#endif

// transpose by 32-bit
void transpose_bin(uint32_t *A, uint32_t *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    //printf("\n n = %d (n mod 32 = %d), m = %d (m mod 32 = %d) \n", n, n % 32, m, m % 32);
    //printf("\n lda = %d (lda mod 32 = %d), ldb = %d (ldb mod 32 = %d) \n", lda, lda % 32, ldb, ldb % 32);
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += 32) {
        int j;
        for (j = 0; j < m; j += 32) {
            int a_index = i*lda + j;
            int b_index = j*ldb + i;
            transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
            //transpose_32x32_bits_my(&A[a_index/32], &B[b_index/32], lda/32, ldb/32);
        }
        for (; j < m; ++j) {
            if (get_bit((const unsigned char* const)A, i * lda + j)) set_bit((unsigned char* const)B, j * ldb + i);
        }
    }
}

static inline int popcnt_32(uint32_t val32) {
#ifdef WIN32  // Windows MSVS
    int tmp_count = __popcnt(val32);
#else   // Linux GCC
    int tmp_count = __builtin_popcount(val32);
#endif
    return tmp_count;
}
//----------------------------

#if (defined(__AVX__) && defined(__x86_64__)) || (defined(_WIN64) && !defined(__MINGW32__))

#if (defined(_WIN64) && !defined(__MINGW64__))
#include <intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#if defined(_MSC_VER) && _MSC_VER <= 1900
static inline __int32 _mm256_extract_epi64(__m256i a, const int index) {
    return a.m256i_i64[index];
}

static inline __int32 _mm256_extract_epi32(__m256i a, const int index) {
    return a.m256i_i32[index];
}
#endif

static inline float _dn_castu32_f32(uint32_t a) {
    return *((float *)&a);
}

static inline float _mm256_extract_float32(__m256 a, const int index) {
    return a.m256_f32[index];
}

#else    // Linux GCC/Clang
#include <x86intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <cpuid.h>

static inline float _dn_castu32_f32(uint32_t a) {
    return *((float *)&a);
}

static inline float _mm256_extract_float32(__m256 a, const int index) {
    switch(index) {
    case 0:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 0));
    case 1:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 1));
    case 2:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 2));
    case 3:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 3));
    case 4:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 4));
    case 5:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 5));
    case 6:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 6));
    case 7:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 7));
    default:
      return _dn_castu32_f32(_mm256_extract_epi32(_mm256_castps_si256(a), 0));
    }
}

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



#ifdef _WIN32
//  Windows
#define cpuid(info, x)    __cpuidex(info, x, 0)
#else
//  GCC Intrinsics
void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif


//  Misc.
static int HW_MMX, HW_x64, HW_RDRAND, HW_BMI1, HW_BMI2, HW_ADX, HW_PREFETCHWT1;
static int HW_ABM;      // Advanced Bit Manipulation

//  SIMD: 128-bit
static int HW_SSE, HW_SSE2, HW_SSE3, HW_SSSE3, HW_SSE41, HW_SSE42, HW_SSE4a, HW_AES, HW_SHA;

//  SIMD: 256-bit
static int HW_AVX, HW_XOP, HW_FMA3, HW_FMA4, HW_AVX2;

//  SIMD: 512-bit
static int HW_AVX512F;    //  AVX512 Foundation
static int HW_AVX512CD;   //  AVX512 Conflict Detection
static int HW_AVX512PF;   //  AVX512 Prefetch
static int HW_AVX512ER;   //  AVX512 Exponential + Reciprocal
static int HW_AVX512VL;   //  AVX512 Vector Length Extensions
static int HW_AVX512BW;   //  AVX512 Byte + Word
static int HW_AVX512DQ;   //  AVX512 Doubleword + Quadword
static int HW_AVX512IFMA; //  AVX512 Integer 52-bit Fused Multiply-Add
static int HW_AVX512VBMI; //  AVX512 Vector Byte Manipulation Instructions

// https://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set
void check_cpu_features(void) {
    int info[4];
    cpuid(info, 0);
    int nIds = info[0];

    cpuid(info, 0x80000000);
    unsigned nExIds = info[0];

    //  Detect Features
    if (nIds >= 0x00000001) {
        cpuid(info, 0x00000001);
        HW_MMX = (info[3] & ((uint32_t)1 << 23)) != 0;
        HW_SSE = (info[3] & ((uint32_t)1 << 25)) != 0;
        HW_SSE2 = (info[3] & ((uint32_t)1 << 26)) != 0;
        HW_SSE3 = (info[2] & ((uint32_t)1 << 0)) != 0;

        HW_SSSE3 = (info[2] & ((uint32_t)1 << 9)) != 0;
        HW_SSE41 = (info[2] & ((uint32_t)1 << 19)) != 0;
        HW_SSE42 = (info[2] & ((uint32_t)1 << 20)) != 0;
        HW_AES = (info[2] & ((uint32_t)1 << 25)) != 0;

        HW_AVX = (info[2] & ((uint32_t)1 << 28)) != 0;
        HW_FMA3 = (info[2] & ((uint32_t)1 << 12)) != 0;

        HW_RDRAND = (info[2] & ((uint32_t)1 << 30)) != 0;
    }
    if (nIds >= 0x00000007) {
        cpuid(info, 0x00000007);
        HW_AVX2 = (info[1] & ((uint32_t)1 << 5)) != 0;

        HW_BMI1 = (info[1] & ((uint32_t)1 << 3)) != 0;
        HW_BMI2 = (info[1] & ((uint32_t)1 << 8)) != 0;
        HW_ADX = (info[1] & ((uint32_t)1 << 19)) != 0;
        HW_SHA = (info[1] & ((uint32_t)1 << 29)) != 0;
        HW_PREFETCHWT1 = (info[2] & ((uint32_t)1 << 0)) != 0;

        HW_AVX512F = (info[1] & ((uint32_t)1 << 16)) != 0;
        HW_AVX512CD = (info[1] & ((uint32_t)1 << 28)) != 0;
        HW_AVX512PF = (info[1] & ((uint32_t)1 << 26)) != 0;
        HW_AVX512ER = (info[1] & ((uint32_t)1 << 27)) != 0;
        HW_AVX512VL = (info[1] & ((uint32_t)1 << 31)) != 0;
        HW_AVX512BW = (info[1] & ((uint32_t)1 << 30)) != 0;
        HW_AVX512DQ = (info[1] & ((uint32_t)1 << 17)) != 0;
        HW_AVX512IFMA = (info[1] & ((uint32_t)1 << 21)) != 0;
        HW_AVX512VBMI = (info[2] & ((uint32_t)1 << 1)) != 0;
    }
    if (nExIds >= 0x80000001) {
        cpuid(info, 0x80000001);
        HW_x64 = (info[3] & ((uint32_t)1 << 29)) != 0;
        HW_ABM = (info[2] & ((uint32_t)1 << 5)) != 0;
        HW_SSE4a = (info[2] & ((uint32_t)1 << 6)) != 0;
        HW_FMA4 = (info[2] & ((uint32_t)1 << 16)) != 0;
        HW_XOP = (info[2] & ((uint32_t)1 << 11)) != 0;
    }
}

int is_avx() {
    static int result = -1;
    if (result == -1) {
        check_cpu_features();
        result = HW_AVX;
        if (result == 1) printf(" Used AVX \n");
        else printf(" Not used AVX \n");
    }
    return result;
}

int is_fma_avx2() {
    static int result = -1;
    if (result == -1) {
        check_cpu_features();
        result = HW_FMA3 && HW_AVX2;
        if (result == 1) printf(" Used FMA & AVX2 \n");
        else printf(" Not used FMA & AVX2 \n");
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
    if (is_avx() == 1) {    // AVX
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
                PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
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



void gemm_nn_fast(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i;

    #pragma omp parallel for
    for (i = 0; i < (M / TILE_M)*TILE_M; i += TILE_M)
    {
        int j, k;
        int i_d, k_d;

        for (k = 0; k < (K / TILE_K)*TILE_K; k += TILE_K)
        {
            for (j = 0; j < (N / TILE_N)*TILE_N; j += TILE_N)
            {
                // L1 - 6 bits tag [11:6] - cache size 32 KB, conflict for each 4 KB
                // L2 - 9 bits tag [14:6] - cache size 256 KB, conflict for each 32 KB
                // L3 - 13 bits tag [18:6] - cache size 8 MB, conflict for each 512 KB

                __m256 result256;
                __m256 a256_0, b256_0;    // AVX
                __m256 a256_1, b256_1;    // AVX
                __m256 a256_2;// , b256_2;    // AVX
                __m256 a256_3;// , b256_3;    // AVX
                __m256 c256_0, c256_1, c256_2, c256_3;
                __m256 c256_4, c256_5, c256_6, c256_7;

                c256_0 = _mm256_loadu_ps(&C[(0 + i)*ldc + (0 + j)]);
                c256_1 = _mm256_loadu_ps(&C[(1 + i)*ldc + (0 + j)]);
                c256_2 = _mm256_loadu_ps(&C[(0 + i)*ldc + (8 + j)]);
                c256_3 = _mm256_loadu_ps(&C[(1 + i)*ldc + (8 + j)]);

                c256_4 = _mm256_loadu_ps(&C[(2 + i)*ldc + (0 + j)]);
                c256_5 = _mm256_loadu_ps(&C[(3 + i)*ldc + (0 + j)]);
                c256_6 = _mm256_loadu_ps(&C[(2 + i)*ldc + (8 + j)]);
                c256_7 = _mm256_loadu_ps(&C[(3 + i)*ldc + (8 + j)]);


                for (k_d = 0; k_d < (TILE_K); ++k_d)
                {
                    a256_0 = _mm256_set1_ps(ALPHA*A[(0 + i)*lda + (k_d + k)]);
                    a256_1 = _mm256_set1_ps(ALPHA*A[(1 + i)*lda + (k_d + k)]);

                    a256_2 = _mm256_set1_ps(ALPHA*A[(2 + i)*lda + (k_d + k)]);
                    a256_3 = _mm256_set1_ps(ALPHA*A[(3 + i)*lda + (k_d + k)]);


                    b256_0 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (0 + j)]);
                    b256_1 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (8 + j)]);

                    // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                    //c256_0 = _mm256_fmadd_ps(a256_0, b256_0, c256_0);
                    //c256_1 = _mm256_fmadd_ps(a256_1, b256_0, c256_1);
                    //c256_2 = _mm256_fmadd_ps(a256_0, b256_1, c256_2);
                    //c256_3 = _mm256_fmadd_ps(a256_1, b256_1, c256_3);

                    //c256_4 = _mm256_fmadd_ps(a256_2, b256_0, c256_4);
                    //c256_5 = _mm256_fmadd_ps(a256_3, b256_0, c256_5);
                    //c256_6 = _mm256_fmadd_ps(a256_2, b256_1, c256_6);
                    //c256_7 = _mm256_fmadd_ps(a256_3, b256_1, c256_7);

                    result256 = _mm256_mul_ps(a256_0, b256_0);
                    c256_0 = _mm256_add_ps(result256, c256_0);

                    result256 = _mm256_mul_ps(a256_1, b256_0);
                    c256_1 = _mm256_add_ps(result256, c256_1);

                    result256 = _mm256_mul_ps(a256_0, b256_1);
                    c256_2 = _mm256_add_ps(result256, c256_2);

                    result256 = _mm256_mul_ps(a256_1, b256_1);
                    c256_3 = _mm256_add_ps(result256, c256_3);


                    result256 = _mm256_mul_ps(a256_2, b256_0);
                    c256_4 = _mm256_add_ps(result256, c256_4);

                    result256 = _mm256_mul_ps(a256_3, b256_0);
                    c256_5 = _mm256_add_ps(result256, c256_5);

                    result256 = _mm256_mul_ps(a256_2, b256_1);
                    c256_6 = _mm256_add_ps(result256, c256_6);

                    result256 = _mm256_mul_ps(a256_3, b256_1);
                    c256_7 = _mm256_add_ps(result256, c256_7);
                }
                _mm256_storeu_ps(&C[(0 + i)*ldc + (0 + j)], c256_0);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (0 + j)], c256_1);
                _mm256_storeu_ps(&C[(0 + i)*ldc + (8 + j)], c256_2);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (8 + j)], c256_3);

                _mm256_storeu_ps(&C[(2 + i)*ldc + (0 + j)], c256_4);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (0 + j)], c256_5);
                _mm256_storeu_ps(&C[(2 + i)*ldc + (8 + j)], c256_6);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (8 + j)], c256_7);
            }

            for (j = (N / TILE_N)*TILE_N; j < N; ++j) {
                for (i_d = i; i_d < (i + TILE_M); ++i_d)
                {
                    for (k_d = k; k_d < (k + TILE_K); ++k_d)
                    {
                        PUT_IN_REGISTER float A_PART = ALPHA*A[i_d*lda + k_d];
                        C[i_d*ldc + j] += A_PART*B[k_d*ldb + j];
                    }
                }
            }
        }

        for (k = (K / TILE_K)*TILE_K; k < K; ++k)
        {
            for (i_d = i; i_d < (i + TILE_M); ++i_d)
            {
                PUT_IN_REGISTER float A_PART = ALPHA*A[i_d*lda + k];
                for (j = 0; j < N; ++j) {
                    C[i_d*ldc + j] += A_PART*B[k*ldb + j];
                }
            }
        }
    }

    for (i = (M / TILE_M)*TILE_M; i < M; ++i) {
        int j, k;
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}



void gemm_nn_bin_32bit_packed(int M, int N, int K, float ALPHA,
    uint32_t *A, int lda,
    uint32_t *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n
        int j, s;
        float mean_val = mean_arr[i];
        //printf(" l.mean_arr[i] = %d \n ", l.mean_arr[i]);
        for (s = 0; s < K; ++s) // l.size*l.size*l.c/32  or (l.size*l.size*l.c)
        {
            PUT_IN_REGISTER uint32_t A_PART = A[i*lda + s];
            __m256i a256 = _mm256_set1_epi32(A_PART);

            for (j = 0; j < N - 8; j += 8)
            {
                __m256i b256 = *((__m256i*)&B[s*ldb + j]);
                __m256i xor256 = _mm256_xor_si256(a256, b256);  // xnor = xor(a,b)
                __m256i all_1 = _mm256_set1_epi8((char)255);
                __m256i xnor256 = _mm256_andnot_si256(xor256, all_1); // xnor = not(xor(a,b))

                // waiting for - CPUID Flags: AVX512VPOPCNTDQ: __m512i _mm512_popcnt_epi32(__m512i a)
                __m256 count = _mm256_setr_ps(
                    popcnt_32(_mm256_extract_epi32(xnor256, 0)),
                    popcnt_32(_mm256_extract_epi32(xnor256, 1)),
                    popcnt_32(_mm256_extract_epi32(xnor256, 2)),
                    popcnt_32(_mm256_extract_epi32(xnor256, 3)),
                    popcnt_32(_mm256_extract_epi32(xnor256, 4)),
                    popcnt_32(_mm256_extract_epi32(xnor256, 5)),
                    popcnt_32(_mm256_extract_epi32(xnor256, 6)),
                    popcnt_32(_mm256_extract_epi32(xnor256, 7)));

                __m256 val2 = _mm256_set1_ps(2);
                count = _mm256_mul_ps(count, val2);     // count * 2

                __m256 val32 = _mm256_set1_ps(32);
                count = _mm256_sub_ps(count, val32);    // count - 32

                __m256 mean256 = _mm256_set1_ps(mean_val);
                count = _mm256_mul_ps(count, mean256);  // count * mean_val

                __m256 c256 = *((__m256*)&C[i*ldc + j]);
                count = _mm256_add_ps(count, c256);     // c = c + count
                *((__m256*)&C[i*ldc + j]) = count;
            }

            for (; j < N; ++j) // out_h*out_w;
            {
                PUT_IN_REGISTER uint32_t B_PART = B[s*ldb + j];
                uint32_t xnor_result = ~(A_PART ^ B_PART);
                int32_t count = popcnt_32(xnor_result);  // must be Signed int

                C[i*ldc + j] += (2 * count - 32) * mean_val;
            }
        }
    }
}

void convolution_2d_old(int w, int h, int ksize, int n, int c, int pad, int stride,
    float *weights, float *input, float *output)
{
    //const int out_h = (h + 2 * pad - ksize) / stride + 1;    // output_height=input_height for stride=1 and pad=1
    //const int out_w = (w + 2 * pad - ksize) / stride + 1;    // output_width=input_width for stride=1 and pad=1

    int fil;
    // filter index
    #pragma omp parallel for      // "omp parallel for" - automatic parallelization of loop by using OpenMP
    for (fil = 0; fil < n; ++fil) {
        //int i, f, j;
        int chan, y, x, f_y, f_x;
        // channel index
        for (chan = 0; chan < c; ++chan)
            // input - y
            for (y = 0; y < h; ++y)
                // input - x
                for (x = 0; x < w; ++x)
                {
                    int const output_index = fil*w*h + y*w + x;
                    int const weights_pre_index = fil*c*ksize*ksize + chan*ksize*ksize;
                    int const input_pre_index = chan*w*h;
                    float sum = 0;

                    // filter - y
                    for (f_y = 0; f_y < ksize; ++f_y)
                    {
                        int input_y = y + f_y - pad;
                        // filter - x
                        for (f_x = 0; f_x < ksize; ++f_x)
                        {
                            int input_x = x + f_x - pad;
                            if (input_y < 0 || input_x < 0 || input_y >= h || input_x >= w) continue;

                            int input_index = input_pre_index + input_y*w + input_x;
                            int weights_index = weights_pre_index + f_y*ksize + f_x;

                            sum += input[input_index] * weights[weights_index];
                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    output[output_index] += sum;
                }
    }
}

void convolution_2d(int w, int h, int ksize, int n, int c, int pad, int stride,
    float *weights, float *input, float *output, float *mean)
{
    //const int out_h = (h + 2 * pad - ksize) / stride + 1;    // output_height=input_height for stride=1 and pad=1
    //const int out_w = (w + 2 * pad - ksize) / stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i;

#if defined(_OPENMP)
    static int max_num_threads = 0;
    if (max_num_threads == 0) {
        max_num_threads = omp_get_max_threads();
        //omp_set_num_threads( max_num_threads / 2);
    }
#endif

    //convolution_2d_old(w, h, ksize, n, c, pad, stride, weights, input, output);

    __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
    for (i = 0; i < ksize*ksize*n*c; i+=8) {
        *((__m256*)&weights[i]) = _mm256_and_ps(*((__m256*)&weights[i]), _mm256_castsi256_ps(all256_sing1));
    }

    //for (i = 0; i < w*h*c; i += 8) {
        //(*(__m256*)&input[i]) = _mm256_and_ps(*((__m256*)&input[i]), _mm256_castsi256_ps(all256_sing1));
    //}


    //__m256i all256_last_zero = _mm256_set1_epi32(0xFFFFFFFF);
    //all256_last_zero.m256i_i32[7] = 0;
    __m256i all256_last_zero =
        _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0);

    __m256i idx256 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
    //__m256 all256_sing1 = _mm256_set1_ps(0x80000000);
    __m256 all256_one = _mm256_set1_ps(1);
    __m256i all256i_one = _mm256_set1_epi32(1);

    ///__m256i src256 = _mm256_loadu_si256((__m256i *)(&src[i]));
    ///__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats

    int fil;
    // filter index
    #pragma omp parallel for      // "omp parallel for" - automatic parallelization of loop by using OpenMP
    for (fil = 0; fil < n; ++fil) {
        int chan, y, x, f_y, f_x;
        float cur_mean = fabs(mean[fil]);
        __m256 mean256 = _mm256_set1_ps(cur_mean);
        // channel index
        //for (chan = 0; chan < c; ++chan)
            // input - y
            for (y = 0; y < h; ++y)
                // input - x
                for (x = 0; x < w-8; x+=8)
                {
                    int const output_index = fil*w*h + y*w + x;
                    float sum = 0;
                    __m256 sum256 = _mm256_set1_ps(0);

                    for (chan = 0; chan < c; ++chan) {
                        int const weights_pre_index = fil*c*ksize*ksize + chan*ksize*ksize;
                        int const input_pre_index = chan*w*h;


                        // filter - y
                        for (f_y = 0; f_y < ksize; ++f_y)
                        {
                            int input_y = y + f_y - pad;
                            //__m256 in = *((__m256*)&input[input_pre_index + input_y*w]);
                            if (input_y < 0 || input_y >= h) continue;
                            //__m256 in = _mm256_loadu_ps(&input[input_pre_index + input_y*w + x - pad]);

                            // filter - x
                            for (f_x = 0; f_x < ksize; ++f_x)
                            {
                                int input_x = x + f_x - pad;
                                //if (input_y < 0 || input_x < 0 || input_y >= h || input_x >= w) continue;

                                int input_index = input_pre_index + input_y*w + input_x;
                                int weights_index = weights_pre_index + f_y*ksize + f_x;
                                //if (input_y < 0 || input_y >= h) continue;

                                //sum += input[input_index] * weights[weights_index];

                                __m256 in = *((__m256*)&input[input_index]);
                                __m256 w = _mm256_set1_ps(weights[weights_index]);
                                //__m256 w_sign = _mm256_and_ps(w, _mm256_castsi256_ps(all256_sing1)); // check sign in 8 x 32-bit floats
                                __m256 xor256 = _mm256_xor_ps(w, in);
                                //printf("\n xor256_1 = %f, xor256_2 = %f \n", xor256.m256_f32[0], xor256.m256_f32[1]);
                                //printf("\n in = %f, w = %f, xor256 = %f \n", in.m256_f32[0], w_sign.m256_f32[0], xor256.m256_f32[0]);

                                //__m256 pn1 = _mm256_and_ps(_mm256_castsi256_ps(all256i_one), xor256);


                                //sum256 = xor256;
                                sum256 = _mm256_add_ps(xor256, sum256);
                                //printf("\n --- \n");
                                //printf("\n 0 = %f, 1 = %f, 2 = %f, 3 = %f, 4 = %f, 5 = %f, 6 = %f, 7 = %f \n", in.m256_f32[0], in.m256_f32[1], in.m256_f32[2], in.m256_f32[3], in.m256_f32[4], in.m256_f32[5], in.m256_f32[6], in.m256_f32[7]);

                                if (f_x < ksize-1) {
                                    //in = _mm256_permutevar8x32_ps(in, idx256);
                                    //in = _mm256_and_ps(in, _mm256_castsi256_ps(all256_last_zero));
                                }
                            }
                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    //output[output_index] += sum;

                    sum256 = _mm256_mul_ps(sum256, mean256);
                    //printf("\n cur_mean = %f, sum256 = %f, sum256 = %f, in = %f \n",
                    //    cur_mean, sum256.m256_f32[0], sum256.m256_f32[1], input[input_pre_index]);

                    //__m256 out = *((__m256*)&output[output_index]);
                    //out = _mm256_add_ps(out, sum256);
                    //(*(__m256*)&output[output_index]) = out;
                    *((__m256*)&output[output_index]) = sum256;

                    //_mm256_storeu_ps(&C[i*ldc + j], result256);
                }
    }
}



// http://graphics.stanford.edu/~seander/bithacks.html
// https://stackoverflow.com/questions/17354971/fast-counting-the-number-of-set-bits-in-m128i-register
// https://arxiv.org/pdf/1611.07612.pdf

static inline int popcnt128(__m128i n) {
    const __m128i n_hi = _mm_unpackhi_epi64(n, n);
#if defined(_MSC_VER)
    return __popcnt64(_mm_cvtsi128_si64(n)) + __popcnt64(_mm_cvtsi128_si64(n_hi));
#elif defined(__APPLE__) && defined(__clang__)
    return _mm_popcnt_u64(_mm_cvtsi128_si64(n)) + _mm_popcnt_u64(_mm_cvtsi128_si64(n_hi));
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
    __m256i val = count256(n);

    //return val.m256i_i64[0] +
    //val.m256i_i64[1] +
    //val.m256i_i64[2] +
    //val.m256i_i64[3];
    return _mm256_extract_epi64(val, 0)
        + _mm256_extract_epi64(val, 1)
        + _mm256_extract_epi64(val, 2)
        + _mm256_extract_epi64(val, 3);
}

static inline void xnor_avx2_popcnt(__m256i a_bit256, __m256i b_bit256, __m256i *count_sum) {
    __m256i c_bit256 = _mm256_set1_epi8((char)255);

    __m256i xor256 = _mm256_xor_si256(a_bit256, b_bit256);  // xnor = not(xor(a,b))
    c_bit256 = _mm256_andnot_si256(xor256, c_bit256);  // can be optimized - we can do other NOT for wegihts once and do not do this NOT

    *count_sum = _mm256_add_epi64(count256(c_bit256), *count_sum);    //  1st part - popcnt Mula's algorithm
}

// 2nd part - popcnt Mula's algorithm
static inline int get_count_mula(__m256i count_sum) {
    return _mm256_extract_epi64(count_sum, 0)
        + _mm256_extract_epi64(count_sum, 1)
        + _mm256_extract_epi64(count_sum, 2)
        + _mm256_extract_epi64(count_sum, 3);
}

// 5x times faster than gemm()-float32
// further optimizations: do mean-mult only for the last layer
void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i;

#if defined(_OPENMP)
    static int max_num_threads = 0;
    if (max_num_threads == 0) {
        max_num_threads = omp_get_max_threads();
        //omp_set_num_threads(max_num_threads / 2);
    }
#endif

    //#pragma omp parallel for
    //for (i = 0; i < M; ++i)
    #pragma omp parallel for
    for (i = 0; i < (M/2)*2; i += 2)
    {   // l.n - filters [16 - 55 - 1024]
        float mean_val_0 = mean_arr[i + 0];
        float mean_val_1 = mean_arr[i + 1];
        int j, k;
        //__m256i all_1 = _mm256_set1_epi8(255);

        //for (j = 0; j < N; ++j)
        for (j = 0; j < (N/2)*2; j += 2)
        { // out_h*out_w - one channel output size [169 - 173056]
            //int count = 0;
            const int bit_step = 256;
            __m256i count_sum_0 = _mm256_set1_epi8(0);
            __m256i count_sum_1 = _mm256_set1_epi8(0);
            __m256i count_sum_2 = _mm256_set1_epi8(0);
            __m256i count_sum_3 = _mm256_set1_epi8(0);

            for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]

                __m256i a_bit256_0 = _mm256_loadu_si256((__m256i *)(A + ((i + 0)*lda + k) / 8));
                __m256i b_bit256_0 = _mm256_loadu_si256((__m256i *)(B + ((j + 0)*ldb + k) / 8));

                __m256i a_bit256_1 = _mm256_loadu_si256((__m256i *)(A + ((i + 1)*lda + k) / 8));
                __m256i b_bit256_1 = _mm256_loadu_si256((__m256i *)(B + ((j + 1)*ldb + k) / 8));


                xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum_0);
                xnor_avx2_popcnt(a_bit256_0, b_bit256_1, &count_sum_1);

                xnor_avx2_popcnt(a_bit256_1, b_bit256_0, &count_sum_2);
                xnor_avx2_popcnt(a_bit256_1, b_bit256_1, &count_sum_3);

                //count += popcnt256(c_bit256);
                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            int count_0 = get_count_mula(count_sum_0);
            int count_1 = get_count_mula(count_sum_1);
            int count_2 = get_count_mula(count_sum_2);
            int count_3 = get_count_mula(count_sum_3);

            const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
            count_0 = count_0 - f1;    // remove extra bits (from empty space for align only)
            count_1 = count_1 - f1;
            count_2 = count_2 - f1;
            count_3 = count_3 - f1;
            C[i*ldc + (j + 0)] = (2 * count_0 - K) * mean_val_0;
            C[i*ldc + (j + 1)] = (2 * count_1 - K) * mean_val_0;
            C[(i + 1)*ldc + (j + 0)] = (2 * count_2 - K) * mean_val_1;
            C[(i + 1)*ldc + (j + 1)] = (2 * count_3 - K) * mean_val_1;
        }

        int i_d;
        for (i_d = 0; i_d < 2; ++i_d)
        {
            float mean_val = mean_arr[i + i_d];
            for (j = (N / 2) * 2; j < N; j += 1)
            { // out_h*out_w - one channel output size [169 - 173056]
                const int bit_step = 256;
                __m256i count_sum = _mm256_set1_epi8(0);

                for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                    __m256i a_bit256_0 = _mm256_loadu_si256((__m256i *)(A + ((i + i_d + 0)*lda + k) / 8));
                    __m256i b_bit256_0 = _mm256_loadu_si256((__m256i *)(B + ((j + 0)*ldb + k) / 8));
                    xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum);
                }
                int count = get_count_mula(count_sum);
                const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    // remove extra bits (from empty space for align only)
                C[(i + i_d)*ldc + j] = (2 * count - K) * mean_val;
            }
        }
    }

    for (i = (M / 2) * 2; i < M; i += 1)
    {
        float mean_val = mean_arr[i];
        int j, k;
        for (j = 0; j < N; j += 1)
        { // out_h*out_w - one channel output size [169 - 173056]
            const int bit_step = 256;
            __m256i count_sum = _mm256_set1_epi8(0);

            for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                __m256i a_bit256_0 = _mm256_loadu_si256((__m256i *)(A + ((i + 0)*lda + k) / 8));
                __m256i b_bit256_0 = _mm256_loadu_si256((__m256i *)(B + ((j + 0)*ldb + k) / 8));
                xnor_avx2_popcnt(a_bit256_0, b_bit256_0, &count_sum);
            }
            int count = get_count_mula(count_sum);
            const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
            count = count - f1;    // remove extra bits (from empty space for align only)
            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}




//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_transpose(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int ldb_align)
{
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;
    int c;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 4; w+=8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = (h * width_col + w)*ldb_align + c;   // transposed & aligned

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    __m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    data_col[col_index + ldb_align * 0] = _mm256_extract_float32(src256, 0);// src256.m256_f32[0];
                    data_col[col_index + ldb_align * 1] = _mm256_extract_float32(src256, 1);// src256.m256_f32[1];
                    data_col[col_index + ldb_align * 2] = _mm256_extract_float32(src256, 2);// src256.m256_f32[2];
                    data_col[col_index + ldb_align * 3] = _mm256_extract_float32(src256, 3);// src256.m256_f32[3];
                    data_col[col_index + ldb_align * 4] = _mm256_extract_float32(src256, 4);// src256.m256_f32[4];
                    data_col[col_index + ldb_align * 5] = _mm256_extract_float32(src256, 5);// src256.m256_f32[5];
                    data_col[col_index + ldb_align * 6] = _mm256_extract_float32(src256, 6);// src256.m256_f32[6];
                    data_col[col_index + ldb_align * 7] = _mm256_extract_float32(src256, 7);// src256.m256_f32[7];

                    //_mm256_storeu_ps(&data_col[col_index], src256);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (h * width_col + w)*ldb_align + c;   // transposed & aligned
                    data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (h * width_col + w)*ldb_align + c;   // transposed & aligned
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (h * width_col + w)*ldb_align + c;   // transposed & aligned
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (h * width_col + w)*ldb_align + c;   // transposed & aligned
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (h * width_col + w)*ldb_align + c;   // transposed & aligned
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }
        }

    }
    else {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = 0; h < height_col; ++h) {
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;

                    int col_index = (h * width_col + w)*ldb_align + c;   // transposed & aligned
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }
        }
    }
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1 && is_fma_avx2())
    {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col-pad; ++h) {
                for (w = pad; w < width_col-pad-8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    __m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    _mm256_storeu_ps(&data_col[col_index], src256);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                w = width_col-1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                            im_row, im_col, c_im, pad);
                }
            }

            {
                h = height_col-1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }
        }

    }
    else {
        //printf("\n Error: is no non-optimized version \n");
        im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
    }
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_align(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1 && is_fma_avx2())
    {
        int new_ldb = bit_align;

        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    __m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    _mm256_storeu_ps(&data_col[col_index], src256);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;
                    data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                }
            }
        }

    }
    else {
        printf("\n Error: is no non-optimized version \n");
        //im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
        // float_to_bit(b, t_input, src_size);
        // transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    }
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_bin(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1 && is_fma_avx2())
    {
        __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
        __m256 float_zero256 = _mm256_set1_ps(0.00);

        int new_ldb = bit_align;

        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //__m256i src256 = _mm256_loadu_si256((__m256i *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    //__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
                    //uint16_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
                    //mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

                    __m256 src256 = _mm256_loadu_ps((float *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    __m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
                    uint16_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

                    uint16_t* dst_ptr = (uint16_t*)&((uint8_t*)data_col)[col_index / 8];
                    *dst_ptr |= (mask << (col_index % 8));
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    float val = data_im[im_col + width*(im_row + height*c_im)];
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char* const)data_col, col_index);
                }
            }
        }

    }
    else {
        printf("\n Error: is no non-optimized version \n");
        //im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
        // float_to_bit(b, t_input, src_size);
        // transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    }
}


void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR)
    {}
    else if (a == LEAKY)
    {
        if (is_fma_avx2()) {
            __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
            __m256 all256_01 = _mm256_set1_ps(0.1F);

            for (i = 0; i < n - 8; i += 8) {
                //x[i] = (x[i]>0) ? x[i] : .1*x[i];

                __m256 src256 = _mm256_loadu_ps(&x[i]);
                __m256 mult256 = _mm256_mul_ps((src256), all256_01); // mult * 0.1

                __m256i sign256 = _mm256_and_si256(_mm256_castps_si256(src256), all256_sing1); // check sign in 8 x 32-bit floats

                __m256 result256 = _mm256_blendv_ps(src256, mult256, _mm256_castsi256_ps(sign256)); // (sign>0) ? src : mult;
                _mm256_storeu_ps(&x[i], result256);
            }
        }

        for (; i < n; ++i) {
            x[i] = (x[i]>0) ? x[i] : .1*x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}

void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    //__m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
    __m256 float_zero256 = _mm256_set1_ps(0.0);

    for (i = 0; i < size; i+=8)
    {
        //__m256i src256 = _mm256_loadu_si256((__m256i *)(&src[i]));
        //__m256i result256 = _mm256_and_si256(src256, all256_sing1); // check sign in 8 x 32-bit floats
        //uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(result256)); // (val >= 0) ? 0 : 1
        ////mask = ~mask;   // inverse mask,  (val >= 0) ? 1 : 0

        __m256 src256 = _mm256_loadu_ps((float *)(&src[i]));
        __m256 result256 = _mm256_cmp_ps(src256, float_zero256, _CMP_GT_OS);
        uint32_t mask = _mm256_movemask_ps(result256); // (val > 0) ? 0 : 1

        dst[i / 8] = mask;
    }
}

static inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb)
{
    __m128 row1 = _mm_loadu_ps(&A[0 * lda]);
    __m128 row2 = _mm_loadu_ps(&A[1 * lda]);
    __m128 row3 = _mm_loadu_ps(&A[2 * lda]);
    __m128 row4 = _mm_loadu_ps(&A[3 * lda]);
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_storeu_ps(&B[0 * ldb], row1);
    _mm_storeu_ps(&B[1 * ldb], row2);
    _mm_storeu_ps(&B[2 * ldb], row3);
    _mm_storeu_ps(&B[3 * ldb], row4);
}

void transpose_block_SSE4x4(float *A, float *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += block_size) {
        int j, i2, j2;
        //int max_i2 = (i + block_size < n) ? (i + block_size) : n;
        if (i + block_size < n) {
            int max_i2 = i + block_size;
            for (j = 0; j < m; j += block_size) {
                //int max_j2 = (j + block_size < m) ? (j + block_size) : m;
                if (j + block_size < m) {
                    int max_j2 = j + block_size;
                    for (i2 = i; i2 < max_i2; i2 += 4) {
                        for (j2 = j; j2 < max_j2; j2 += 4) {
                            transpose4x4_SSE(&A[i2*lda + j2], &B[j2*ldb + i2], lda, ldb);
                        }
                    }
                }
                else {
                    for (i2 = i; i2 < max_i2; ++i2) {
                        for (j2 = j; j2 < m; ++j2) {
                            B[j2*ldb + i2] = A[i2*lda + j2];
                        }
                    }
                }
            }
        }
        else {
            for (i2 = i; i2 < n; ++i2) {
                for (j2 = 0; j2 < m; ++j2) {
                    B[j2*ldb + i2] = A[i2*lda + j2];
                }
            }
        }
    }
}


void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{

    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;
    int b, k;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                //for (j = 0; j < out_w; ++j) {
                j = 0;

                if(stride == 1 && is_avx() == 1) {
                    for (j = 0; j < out_w - 8 - (size - 1); j += 8) {
                        int out_index = j + out_w*(i + out_h*(k + c*b));
                        __m256 max256 = _mm256_set1_ps(-FLT_MAX);
                        for (n = 0; n < size; ++n) {
                            for (m = 0; m < size; ++m) {
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + w*(cur_h + h*(k + b*c));
                                int valid = (cur_h >= 0 && cur_h < h &&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                max256 = _mm256_max_ps(src256, max256);
                            }
                        }
                        _mm256_storeu_ps(&dst[out_index], max256);

                    }
                }
                else if (size == 2 && stride == 2 && is_avx() == 1) {
                    for (j = 0; j < out_w - 4; j += 4) {
                        int out_index = j + out_w*(i + out_h*(k + c*b));
                        //float max = -FLT_MAX;
                        //int max_i = -1;
                        __m128 max128 = _mm_set1_ps(-FLT_MAX);

                        for (n = 0; n < size; ++n) {
                            //for (m = 0; m < size; ++m)
                            m = 0;
                            {
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + w*(cur_h + h*(k + b*c));
                                int valid = (cur_h >= 0 && cur_h < h &&
                                    cur_w >= 0 && cur_w < w);
                                if (!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);
                                __m256 src256_2 = _mm256_permute_ps(src256, (1 << 0) | (3 << 4));
                                __m256 max256 = _mm256_max_ps(src256, src256_2);

                                __m128 src128_0 = _mm256_extractf128_ps(max256, 0);
                                __m128 src128_1 = _mm256_extractf128_ps(max256, 1);
                                __m128 src128 = _mm_shuffle_ps(src128_0, src128_1, (2 << 2) | (2 << 6));

                                max128 = _mm_max_ps(src128, max128);
                            }
                        }
                        _mm_storeu_ps(&dst[out_index], max128);
                    }
                }

                for (; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    if (indexes) indexes[out_index] = max_i;
                }
            }
        }
    }
}

#else   // AVX

int is_avx() {
    return 0;
}

int is_fma_avx2() {
    return 0;
}

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void gemm_nn_fast(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    #pragma omp parallel for
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void gemm_nn_bin_32bit_packed(int M, int N, int K, float ALPHA,
    uint32_t *A, int lda,
    uint32_t *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n
        int j, s;
        float mean_val = mean_arr[i];
        //printf(" l.mean_arr[i] = %d \n ", l.mean_arr[i]);
        for (s = 0; s < K; ++s) // l.size*l.size*l.c/32  or (l.size*l.size*l.c)
        {
            //PUT_IN_REGISTER float A_PART = 1*a[i*k + s];
            PUT_IN_REGISTER uint32_t A_PART = A[i * lda + s];
            for (j = 0; j < N; ++j) // out_h*out_w;
            {
                //c[i*n + j] += A_PART*b[s*n + j];
                PUT_IN_REGISTER uint32_t B_PART = B[s * ldb + j];
                uint32_t xnor_result = ~(A_PART ^ B_PART);
                //printf(" xnor_result = %d, ", xnor_result);
                int32_t count = popcnt_32(xnor_result);  // must be Signed int

                C[i*ldc + j] += (2 * count - 32) * mean_val;
                //c[i*n + j] += count*mean;
            }
        }
    }
}


void convolution_2d(int w, int h, int ksize, int n, int c, int pad, int stride,
    float *weights, float *input, float *output, float *mean)
{
    const int out_h = (h + 2 * pad - ksize) / stride + 1;    // output_height=input_height for stride=1 and pad=1
    const int out_w = (w + 2 * pad - ksize) / stride + 1;    // output_width=input_width for stride=1 and pad=1
    //int i, f, j;

    int fil;
    // filter index
    #pragma omp parallel for      // "omp parallel for" - automatic parallelization of loop by using OpenMP
    for (fil = 0; fil < n; ++fil) {
        int chan, y, x, f_y, f_x;
        // channel index
        for (chan = 0; chan < c; ++chan)
            // input - y
            for (y = 0; y < h; ++y)
                // input - x
                for (x = 0; x < w; ++x)
                {
                    int const output_index = fil*w*h + y*w + x;
                    int const weights_pre_index = fil*c*ksize*ksize + chan*ksize*ksize;
                    int const input_pre_index = chan*w*h;
                    float sum = 0;

                    // filter - y
                    for (f_y = 0; f_y < ksize; ++f_y)
                    {
                        int input_y = y + f_y - pad;
                        // filter - x
                        for (f_x = 0; f_x < ksize; ++f_x)
                        {
                            int input_x = x + f_x - pad;
                            if (input_y < 0 || input_x < 0 || input_y >= h || input_x >= w) continue;

                            int input_index = input_pre_index + input_y*w + input_x;
                            int weights_index = weights_pre_index + f_y*ksize + f_x;

                            sum += input[input_index] * weights[weights_index];
                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    output[output_index] += sum;
                }
    }
}

static inline int popcnt_64(uint64_t val64) {
#ifdef WIN32  // Windows
#ifdef _WIN64 // Windows 64-bit
    int tmp_count = __popcnt64(val64);
#else         // Windows 32-bit
    int tmp_count = __popcnt(val64);
    tmp_count += __popcnt(val64 >> 32);
#endif
#else   // Linux
#if defined(__x86_64__) || defined(__aarch64__)  // Linux 64-bit
    int tmp_count = __builtin_popcountll(val64);
#else  // Linux 32-bit
    int tmp_count = __builtin_popcount(val64);
    tmp_count += __builtin_popcount(val64 >> 32);
#endif
#endif
    return tmp_count;
}

void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
    unsigned char *A, int lda,
    unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i;

    #pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
        int j, k;
        float mean_val = mean_arr[i];

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));
                uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));
                uint64_t c_bit64 = xnor_int64(a_bit64, b_bit64);

                int tmp_count = popcnt_64(c_bit64);

                if (K - k < 64)  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
                count += tmp_count;
                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}

void im2col_cpu_custom_transpose(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int ldb_align)
{
    printf("\n im2col_cpu_custom_transpose() isn't implemented without AVX \n");
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
    return;

    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                }
    }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
                }
            }
        }

    }
    else {
        //printf("\n Error: is no non-optimized version \n");
        im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
    }
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_bin(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int bit_align)
{
    int c;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        int new_ldb = bit_align;

        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int h, w;
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col - pad; ++h) {
                for (w = pad; w < width_col - pad - 8; w += 1) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    float val = data_im[im_col + width*(im_row + height*c_im)];
                    if (val > 0) set_bit((unsigned char*)data_col, col_index);
                }

                for (; w < width_col - pad; ++w) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    float val = data_im[im_col + width*(im_row + height*c_im)];
                    if (val > 0) set_bit((unsigned char*)data_col, col_index);
                }
            }

            {
                w = 0;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char*)data_col, col_index);
                }
            }

            {
                w = width_col - 1;
                for (h = 0; h < height_col; ++h) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char*)data_col, col_index);
                }
            }

            {
                h = 0;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char*)data_col, col_index);
                }
            }

            {
                h = height_col - 1;
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h;
                    int im_col = w_offset + w;
                    //int col_index = (c * height_col + h) * width_col + w;
                    int col_index = c * new_ldb + h * width_col + w;

                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    float val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                    if (val > 0) set_bit((unsigned char*)data_col, col_index);
                }
            }
        }

    }
    else {
        printf("\n Error: is no non-optimized version \n");
        //im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col); // must be aligned for transpose after float_to_bin
        // float_to_bit(b, t_input, src_size);
        // transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    }
}


void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR)
    {
    }
    else if (a == LEAKY)
    {
        for (i = 0; i < n; ++i) {
            x[i] = (x[i]>0) ? x[i] : .1*x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}

void float_to_bit(float *src, unsigned char *dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);

    size_t i;
    char* byte_arr = (char*)xcalloc(size, sizeof(char));
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

static inline void transpose_scalar_block(float *A, float *B, const int lda, const int ldb, const int block_size)
{
    int i;
    //#pragma omp parallel for
    for (i = 0; i<block_size; i++) {
        int j;
        for (j = 0; j<block_size; j++) {
            B[j*ldb + i] = A[i*lda + j];
        }
    }
}

void transpose_block_SSE4x4(float *A, float *B, const int n, const int m,
    const int lda, const int ldb, const int block_size)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i += block_size) {
        int j, i2, j2;
        for (j = 0; j < m; j += block_size) {
            int max_i2 = i + block_size < n ? i + block_size : n;
            int max_j2 = j + block_size < m ? j + block_size : m;
            for (i2 = i; i2 < max_i2; ++i2) {
                for (j2 = j; j2 < max_j2; ++j2) {
                    B[j2*ldb + i2] = A[i2*lda + j2];
                }
                }
            }
        }
}

void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{
    int b, k;
    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                for (j = 0; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    if (indexes) indexes[out_index] = max_i;
                }
            }
        }
    }
}

#endif    // AVX


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void repack_input(float *input, float *re_packed_input, int w, int h, int c)
{
    const int items_per_channel = w * h;
    int chan, i;
    for (chan = 0; chan < c; chan += 32)
    {
        for (i = 0; i < items_per_channel; ++i)
        {
            int c_pack;
            for (c_pack = 0; c_pack < 32; ++c_pack) {
                float src = input[(chan + c_pack)*items_per_channel + i];

                re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
            }
        }
    }
}

void transpose_uint32(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
    //l.bit_align - algined (n) by 32
    //new_ldb - aligned (k) by 256

    int i;
    //#pragma omp parallel for
    for (i = 0; i < src_h; i += 1)  // l.size*l.size*l.c;
    {
        int j;
        for (j = 0; j < src_w; j += 1)  // out_h*out_w;
        {
            ((uint32_t *)dst)[j*dst_align / 32 + i] = ((uint32_t *)src)[i*src_align + j];
        }
    }
}

void gemm_nn_bin_transposed_32bit_packed(int M, int N, int K, float ALPHA,
    uint32_t *A, int lda,
    uint32_t *B, int ldb,
    float *C, int ldc, float *mean_arr)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < M; ++i) {   // l.n
        int j, s;
        float mean_val = mean_arr[i];
        for (j = 0; j < N; ++j) // out_h*out_w;
        {
            float val = 0;
            for (s = 0; s < K; ++s) // l.size*l.size*l.c/32  or (l.size*l.size*l.c)
            {
                PUT_IN_REGISTER uint32_t A_PART = ((uint32_t*)A)[i*lda + s];
                PUT_IN_REGISTER uint32_t B_PART = ((uint32_t*)B)[j * ldb + s];
                uint32_t xnor_result = ~(A_PART ^ B_PART);
                int32_t count = popcnt_32(xnor_result);  // must be Signed int

                val += (2 * count - 32) * mean_val;
            }
            C[i*ldc + j] += val;
        }
    }
}

void convolution_repacked(uint32_t *packed_input, uint32_t *packed_weights, float *output,
    int w, int h, int c, int n, int size, int pad, int new_lda, float *mean_arr)
{
    int fil;
    // filter index
    #pragma omp parallel for
    for (fil = 0; fil < n; ++fil) {
        float mean_val = mean_arr[fil];
        int chan, y, x, f_y, f_x;   // c_pack
        // channel index
        for (chan = 0; chan < c / 32; ++chan)
            //for (chan = 0; chan < l.c; chan += 32)
            //for (c_pack = 0; c_pack < 32; ++c_pack)
            // input - y
            for (y = 0; y < h; ++y)
                // input - x
                for (x = 0; x < w; ++x)
                {
                    int const output_index = fil*w*h + y*w + x;
                    float sum = 0;

                    // filter - y
                    for (f_y = 0; f_y < size; ++f_y)
                    {
                        int input_y = y + f_y - pad;
                        // filter - x
                        for (f_x = 0; f_x < size; ++f_x)
                        {
                            int input_x = x + f_x - pad;
                            if (input_y < 0 || input_x < 0 || input_y >= h || input_x >= w) continue;

                            // normal
                            //float input = state.input[(chan + c_pack)*l.w*l.h + input_y*l.w + input_x];
                            //float weight = l.weights[fil*l.c*l.size*l.size + (chan + c_pack)*l.size*l.size + f_y*l.size + f_x];

                            // packed
                            //float input = re_packed_input[chan*l.w*l.h + (input_y*l.w + input_x) * 32 + c_pack];
                            //float weight = l.weights[fil*l.c*l.size*l.size + chan*l.size*l.size + (f_y*l.size + f_x) * 32 + c_pack];
                            //sum += input * weight;

                            //float input = re_packed_input[chan*l.w*l.h + (input_y*l.w + input_x) * 32 + c_pack];
                            //float weight = l.weights[fil*l.c*l.size*l.size + chan*l.size*l.size + (f_y*l.size + f_x) * 32 + c_pack];
                            //uint32_t bit1 = input > 0;
                            //uint32_t bit2 = weight > 0;
                            //uint32_t count = (~(bit1 ^ bit2)) & 1;
                            //float result = (2 * (float)count - 1) * mean_val;
                            //printf("\n mul = %f, bit1 = %d, bit2 = %d, count = %d, mean = %f, result = %f  ", input*weight, bit1, bit2, count, mean_val, result);
                            //sum += result;

                            uint32_t input = ((uint32_t *)packed_input)[chan*w*h + input_y*w + input_x];
                            //uint32_t weight = ((uint32_t *)l.align_bit_weights)[fil*l.c*l.size*l.size/32 + chan*l.size*l.size + f_y*l.size + f_x];
                            uint32_t weight = ((uint32_t *)packed_weights)[fil*new_lda / 32 + chan*size*size + f_y*size + f_x];

                            uint32_t xnor_result = ~(input ^ weight);
                            int32_t count = popcnt_32(xnor_result); // mandatory Signed int
                            sum += (2 * count - 32) * mean_val;
                        }
                    }
                    // l.output[filters][width][height] +=
                    //        state.input[channels][width][height] *
                    //        l.weights[filters][channels][filter_width][filter_height];
                    output[output_index] += sum;
                }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            PUT_IN_REGISTER float sum = 0;
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
            PUT_IN_REGISTER float A_PART = ALPHA * A[k * lda + i];
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
            PUT_IN_REGISTER float sum = 0;
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

    is_avx();   // initialize static variable
    if (is_fma_avx2() && !TA && !TB) {
        gemm_nn_fast(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else {
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
    cudaError_t stream_status = (cudaError_t)cublasSetStream(handle, get_cuda_stream());
    CHECK_CUDA(stream_status);
    cudaError_t status = (cudaError_t)cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    CHECK_CUDA(status);
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
        cudaDeviceSynchronize();
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



void init_cpu() {
    is_avx();
    is_fma_avx2();
}
