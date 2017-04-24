int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include "assert.h"
#include <stdlib.h>
#include <time.h>



void cuda_set_device(int n)
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
#ifdef __linux__
        snprintf(buffer, 256, "CUDA Error: %s", s);
#else
		_snprintf(buffer, 256, "CUDA Error: %s", s);
#endif
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
#ifdef __linux__
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
#else
		_snprintf(buffer, 256, "CUDA Error Prev: %s", s);
#endif
        error(buffer);
    } 
}

void check_cublas_error(cublasStatus_t status)
{
    const char *s;
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: 
		s = "CUBLAS_STATUS_SUCCESS";
		return;
        case CUBLAS_STATUS_NOT_INITIALIZED: 
		s = "CUBLAS_STATUS_NOT_INITIALIZED";
		break;
        case CUBLAS_STATUS_ALLOC_FAILED: 
		s = "CUBLAS_STATUS_ALLOC_FAILED";
		break;
        case CUBLAS_STATUS_INVALID_VALUE: 
		s = "CUBLAS_STATUS_INVALID_VALUE";
		break; 
        case CUBLAS_STATUS_ARCH_MISMATCH: 
		s = "CUBLAS_STATUS_ARCH_MISMATCH";
		break; 
        case CUBLAS_STATUS_MAPPING_ERROR: 
		s = "CUBLAS_STATUS_MAPPING_ERROR";
		break;
        case CUBLAS_STATUS_EXECUTION_FAILED: 
		s = "CUBLAS_STATUS_EXECUTION_FAILED";
		break; 
        case CUBLAS_STATUS_INTERNAL_ERROR: 
		s = "CUBLAS_STATUS_INTERNAL_ERROR";
		break; 
   	default:
    		s = "CUBLAS unknown error";
    }

	char buffer[256];
	printf("CUBLAS Error : %s, value = %d\n", s, status);
	assert(0);
#ifdef __linux__
	snprintf(buffer, 256, "CUBLAS Error Prev: %s", s);
#else
	_snprintf(buffer, 256, "CUBLAS Error Prev: %s", s);
#endif
	error(buffer);
}


dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt((long double)k));
        y = (n-1)/(x*BLOCK) + 1;
    }
#ifdef __cplusplus    
    dim3 d (x, y, 1);
#else
    dim3 d = {x, y, 1};
#endif
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN

#define MAX_CUDNN (16)
static int cudnn_init[MAX_CUDNN] = {0};
static cudnnHandle_t cudnn_handle_t[MAX_CUDNN];
cudnnHandle_t cudnn_handle()
{
    int i = cuda_get_device();
    if(!cudnn_init[i]) {
        cudnnCreate(&cudnn_handle_t[i]);
        cudnn_init[i] = 1;
    }
    return cudnn_handle_t[i];
}

void cudnn_handle_reset()
{
    for(int i = 0;i < MAX_CUDNN;i ++) 
    {
        cudnn_init[i] = 0;
        cudnn_handle_t[i] = 0;
    }
}

#endif

#define MAX_BLAS (16)
static int blas_init[MAX_BLAS] = {0};
static cublasHandle_t blas_handle_t[MAX_BLAS];
void blas_handle_reset()
{
    for(int i = 0;i < MAX_BLAS;i ++) 
    {
        blas_init[i] = 0;
        blas_handle_t[i] = 0;
    }
}

cublasHandle_t blas_handle()
{
    int i = cuda_get_device();
    if(!blas_init[i]) {
        cublasCreate(&blas_handle_t[i]);
        blas_init[i] = 1;
    }
    return blas_handle_t[i];
}


#ifdef _ENABLE_CUDA_MEM_DEBUG
void cuda_dump_mem_stat()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("CUDA Memory Status: Free/Total = [%lu]/[%lu]\n", free, total);
}

#endif


#ifdef _ENABLE_CUDA_MEM_DEBUG
// For debugging CUDA allocations
static unsigned int cuda_make_array_cnt = 0;
static unsigned int cuda_free_cnt = 0;
static unsigned long long cuda_make_array_size_float = 0;
static unsigned long long cuda_make_array_size_int = 0;
#endif
float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
#ifdef _ENABLE_CUDA_MEM_DEBUG  
    printf("CUDA alloc/free cnts/size/reqsize = [%d], [%d], [%llu], [%lu]\n", cuda_make_array_cnt, cuda_free_cnt, cuda_make_array_size_float, n ); 
#endif   
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
#ifdef _ENABLE_CUDA_MEM_DEBUG    
    else
    {
        float* cptr = (float*) calloc(size, 1);
        if(cptr) {
            cudaMemcpy(x_gpu, cptr, size,cudaMemcpyHostToDevice);        
            free(cptr);
        }
    }    
    cuda_make_array_cnt ++;
    cuda_make_array_size_float += n;
    printf("cuda_make_array allocated [%p] of [%lu]\n", x_gpu, size);
#endif    
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]){
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = (float*)calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
#ifdef _ENABLE_CUDA_MEM_DEBUG    
    printf("CUDA alloc/free cnts/size/reqsize(int) = [%d], [%d], [%llu], [%lu]\n", cuda_make_array_cnt, cuda_free_cnt, cuda_make_array_size_int, n );        
    if(x_gpu)
    {
        float* cptr = (float*) calloc(size, 1);
        if(cptr) {
            cudaMemcpy(x_gpu, cptr, size,cudaMemcpyHostToDevice);        
            free(cptr);
        }
    }
    cuda_make_array_cnt ++;
    cuda_make_array_size_int += n;
    printf("cuda_make_int_array allocated [%p] of [%lu]\n", x_gpu, size);    
#endif    
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    if(!x_gpu)
    {
        printf("cuda_free called with nil x_gpu\n");
        return;
    }
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
#ifdef _ENABLE_CUDA_MEM_DEBUG       
    cuda_free_cnt ++;
    printf("cuda_free freed [%p]\n", x_gpu);
#endif    
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    if(!x_gpu)
    {
        printf("cuda_push_array called with nil x_gpu\n");
        return;
    }    
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    if(!x_gpu || !x)
    {
        printf("cuda_pull_array called with nil x_gpu or nil x\n");
        return;
    }    
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = (float*)calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    if(temp) free(temp);
    return m;
}

#endif
