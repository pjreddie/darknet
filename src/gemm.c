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

#if (defined(__AVX__) && defined(__x86_64__)) || defined(_WIN64)

#define OSXSAVEFlag (1UL<<27)
#define AVXFlag     ((1UL<<28)|OSXSAVEFlag)
#define FMAFlag     ((1UL<<12)|AVXFlag|OSXSAVEFlag)
#define CLMULFlag   ((1UL<< 1)|AVXFlag|OSXSAVEFlag)
#define VAESFlag    ((1UL<<25)|AVXFlag|OSXSAVEFlag)

#include <stdint.h>

#ifdef _WIN64
#include <intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#else	// Linux GCC/Clang
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
	uint32_t regs[4];	// EAX, EBX, ECX, EDX;
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
	if (is_fma_avx() == 1) {	// AVX
		for (i = 0; i < M; ++i) {
			for (k = 0; k < K; ++k) {
				float A_PART = ALPHA*A[i*lda + k];
				__m256 a256, b256, c256, result256;	// AVX
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
				__m128 a128, b128, c128, result128;	// SSE
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
#endif	// __x86_64

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
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
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

