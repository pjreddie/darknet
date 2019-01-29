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
        snprintf(buffer, 256, "CUDA Error: %s", s);
#ifdef WIN32
        getchar();
#endif
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status2);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
#ifdef WIN32
        getchar();
#endif
        error(buffer);
    }
}

void check_error_extended(cudaError_t status, char *file, int line, char *date_time)
{
    if (status != cudaSuccess)
        printf("CUDA Error: file: %s() : line: %d : build time: %s \n", file, line, date_time);
    check_error(status);
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

static cudaStream_t streamsArray[16];    // cudaStreamSynchronize( get_cuda_stream() );
static int streamInit[16] = { 0 };

cudaStream_t get_cuda_stream() {
    int i = cuda_get_device();
    if (!streamInit[i]) {
        cudaError_t status = cudaStreamCreate(&streamsArray[i]);
        //cudaError_t status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            printf(" cudaStreamCreate error: %d \n", status);
            const char *s = cudaGetErrorString(status);
            char buffer[256];
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamDefault);
            check_error(status);
        }
        streamInit[i] = 1;
    }
    return streamsArray[i];
}

static cudaStream_t streamsArray2[16];    // cudaStreamSynchronize( get_cuda_memcpy_stream() );
static int streamInit2[16] = { 0 };

cudaStream_t get_cuda_memcpy_stream() {
    int i = cuda_get_device();
    if (!streamInit2[i]) {
        cudaError_t status = cudaStreamCreate(&streamsArray2[i]);
        //cudaError_t status = cudaStreamCreateWithFlags(&streamsArray2[i], cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            printf(" cudaStreamCreate-Memcpy error: %d \n", status);
            const char *s = cudaGetErrorString(status);
            char buffer[256];
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArray2[i], cudaStreamDefault);
            check_error(status);
        }
        streamInit2[i] = 1;
    }
    return streamsArray2[i];
}


#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
        cudnnStatus_t status = cudnnSetStream(handle[i], get_cuda_stream());
    }
    return handle[i];
}


void cudnn_check_error(cudnnStatus_t status)
{
    //cudaDeviceSynchronize();
    cudnnStatus_t status2 = CUDNN_STATUS_SUCCESS;
#ifdef CUDNN_ERRQUERY_RAWCODE
    cudnnStatus_t status_tmp = cudnnQueryRuntimeError(cudnn_handle(), &status2, CUDNN_ERRQUERY_RAWCODE, NULL);
#endif
    if (status != CUDNN_STATUS_SUCCESS)
    {
        const char *s = cudnnGetErrorString(status);
        char buffer[256];
        printf("cuDNN Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "cuDNN Error: %s", s);
#ifdef WIN32
        getchar();
#endif
        error(buffer);
    }
    if (status2 != CUDNN_STATUS_SUCCESS)
    {
        const char *s = cudnnGetErrorString(status2);
        char buffer[256];
        printf("cuDNN Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "cuDNN Error Prev: %s", s);
#ifdef WIN32
        getchar();
#endif
        error(buffer);
    }
}

void cudnn_check_error_extended(cudnnStatus_t status, char *file, int line, char *date_time)
{
    if (status != cudaSuccess)
        printf("\n cuDNN Error in: file: %s() : line: %d : build time: %s \n", file, line, date_time);
    cudnn_check_error(status);
}
#endif

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        cublasStatus_t status = cublasSetStream(handle[i], get_cuda_stream());
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    if (status != cudaSuccess) fprintf(stderr, " Try to set subdivisions=64 in your cfg-file. \n");
    check_error(status);
    if(x){
        //status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, get_cuda_stream());
        check_error(status);
    }
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
    float *tmp = calloc(n, sizeof(float));
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
    if(status != cudaSuccess) fprintf(stderr, " Try to set subdivisions=64 in your cfg-file. \n");
    check_error(status);
    return x_gpu;
}

int *cuda_make_int_array_new_api(int *x, size_t n)
{
	int *x_gpu;
	size_t size = sizeof(int)*n;
	cudaError_t status = cudaMalloc((void **)&x_gpu, size);
	check_error(status);
	if (x) {
		//status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice, get_cuda_stream());
        cudaError_t status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, get_cuda_stream());
		check_error(status);
	}
	if (!x_gpu) error("Cuda malloc failed\n");
	return x_gpu;
}

void cuda_free(float *x_gpu)
{
    //cudaStreamSynchronize(get_cuda_stream());
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    //cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    cudaError_t status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, get_cuda_stream());
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    //cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    cudaError_t status = cudaMemcpyAsync(x, x_gpu, size, cudaMemcpyDeviceToHost, get_cuda_stream());
    check_error(status);
    cudaStreamSynchronize(get_cuda_stream());
}

void cuda_pull_array_async(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpyAsync(x, x_gpu, size, cudaMemcpyDeviceToHost, get_cuda_stream());
    check_error(status);
    //cudaStreamSynchronize(get_cuda_stream());
}

int get_number_of_blocks(int array_size, int block_size)
{
    return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
}

#else // GPU
#include "cuda.h"
void cuda_set_device(int n) {}
#endif // GPU
