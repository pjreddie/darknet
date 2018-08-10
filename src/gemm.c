#include "gemm.h"
#include "utils.h"
#include "im2col.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
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


void convolution_2d(int w, int h, int ksize, int n, int c, int pad, int stride,
    float *weights, float *input, float *output)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (w + 2 * pad - ksize) / stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;

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



// http://graphics.stanford.edu/~seander/bithacks.html
// https://stackoverflow.com/questions/17354971/fast-counting-the-number-of-set-bits-in-m128i-register
// https://arxiv.org/pdf/1611.07612.pdf

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
    __m256i val = count256(n);

    return val.m256i_i64[0] +
    val.m256i_i64[1] +
    val.m256i_i64[2] +
    val.m256i_i64[3];
}

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
        omp_set_num_threads(max_num_threads / 2);
    }
#endif

    #pragma omp parallel for
    for (i = 0; i < M; ++i)
    {   // l.n - filters [16 - 55 - 1024]
        float mean_val = mean_arr[i];
        int j, k;
        __m256i all_1 = _mm256_set1_epi8(255);

        for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;
            const int bit_step = 256;
            __m256i count_sum = _mm256_set1_epi8(0);

            for (k = 0; k < K; k += bit_step) {   // l.size*l.size*l.c - one filter size [27 - 9216]
                __m256i a_bit256 = _mm256_loadu_si256((__m256i *)(A + (i*lda + k) / 8));
                __m256i b_bit256 = _mm256_loadu_si256((__m256i *)(B + (j*ldb + k) / 8));
                __m256i xor256 = _mm256_xor_si256(a_bit256, b_bit256);  // xnor = not(xor(a,b))
                __m256i c_bit256 = _mm256_andnot_si256(xor256, all_1);  // can be optimized - we can do other NOT for wegihts once and do not do this NOT

                count_sum = _mm256_add_epi64(count256(c_bit256), count_sum);    //  Mula’s algorithm

                //count += popcnt256(c_bit256);

                //binary_int64_printf(c_bit64);
                //printf(", count = %d \n\n", tmp_count);
            }

            // count of 1 bits
            count = count_sum.m256i_i64[0] +
                count_sum.m256i_i64[1] +
                count_sum.m256i_i64[2] +
                count_sum.m256i_i64[3];

            int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
            count = count - f1;    // remove extra bits (from empty space for align only)

            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}


static inline float im2col_get_pixel(float *im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_custom_transpose(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col, int ldb_align)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
#pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
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
                    __m256 src256 = _mm256_loadu_ps((__m256i *)(&data_im[im_col + width*(im_row + height*c_im)]));
                    data_col[col_index + ldb_align * 0] = src256.m256_f32[0];
                    data_col[col_index + ldb_align * 1] = src256.m256_f32[1];
                    data_col[col_index + ldb_align * 2] = src256.m256_f32[2];
                    data_col[col_index + ldb_align * 3] = src256.m256_f32[3];
                    data_col[col_index + ldb_align * 4] = src256.m256_f32[4];
                    data_col[col_index + ldb_align * 5] = src256.m256_f32[5];
                    data_col[col_index + ldb_align * 6] = src256.m256_f32[6];
                    data_col[col_index + ldb_align * 7] = src256.m256_f32[7];

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

    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = pad; h < height_col-pad; ++h) {
                for (w = pad; w < width_col-pad-8; w += 8) {
                    int im_row = h_offset + h - pad;
                    int im_col = w_offset + w - pad;
                    int col_index = (c * height_col + h) * width_col + w;

                    //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                    __m256 src256 = _mm256_loadu_ps((__m256i *)(&data_im[im_col + width*(im_row + height*c_im)]));
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

void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR)
    {}
    else if (a == LEAKY)
    {
        __m256i all256_sing1 = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
        __m256 all256_01 = _mm256_set1_ps(0.1F);

        for (i = 0; i < n-8; i += 8) {
            //x[i] = (x[i]>0) ? x[i] : .1*x[i];

            __m256 src256 = _mm256_loadu_ps((__m256 *)(&x[i]));
            __m256 mult256 = _mm256_mul_ps((src256), all256_01); // mult * 0.1

            __m256i sign256 = _mm256_and_si256(_mm256_castps_si256(src256), all256_sing1); // check sign in 8 x 32-bit floats

            __m256 result256 = _mm256_blendv_ps(src256, mult256, _mm256_castsi256_ps(sign256)); // (sign>0) ? src : mult;
            _mm256_storeu_ps((__m256 *)(&x[i]), result256);
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


void convolution_2d(int w, int h, int ksize, int n, int c, int pad, int stride,
    float *weights, float *input, float *output)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (w + 2 * pad - ksize) / stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;

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

    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    // optimized version
    if (height_col == height && width_col == width && stride == 1 && pad == 1)
    {
        #pragma omp parallel for
        for (c = 0; c < channels_col; ++c) {
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

static inline void transpose_scalar_block(float *A, float *B, const int lda, const int ldb, const int block_size)
{
    int i, j;
    //#pragma omp parallel for
    for (i = 0; i<block_size; i++) {
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

