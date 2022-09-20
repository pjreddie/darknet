int gpu_index = 0;

#ifdef GPU

#include "hip/hip_runtime.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

void cuda_set_device(int n) {
    gpu_index = n;
    hipError_t status = hipSetDevice(n);
    check_error(status);
}

int cuda_get_device() {
    int n = 0;
    hipError_t status = hipGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(hipError_t status) {
    //hipDeviceSynchronize();
    hipError_t status2 = hipGetLastError();
    if (status != hipSuccess) {
        const char *s = hipGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != hipSuccess) {
        const char *s = hipGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

dim3 cuda_gridsize(size_t n) {
    size_t k = (n - 1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d = {(uint32_t)x, (uint32_t)y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
hipdnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static hipdnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        hipdnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

hipblasHandle_t blas_handle() {
    static int init[16] = {0};
    static hipblasHandle_t handle[16];
    int i = cuda_get_device();
    if (!init[i]) {
        hipblasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n) {
    float *x_gpu;
    size_t size = sizeof(float) * n;
    hipError_t status = hipMalloc((void **) &x_gpu, size);
    check_error(status);
    if (x) {
        status = hipMemcpy(x_gpu, x, size, hipMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if (!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n) {
    static hiprandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if (!init[i]) {
        hiprandCreateGenerator(&gen[i], HIPRAND_RNG_PSEUDO_DEFAULT);
        hiprandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    hiprandGenerateUniform(gen[i], x_gpu, n);
    check_error(hipPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s) {
    float *tmp = (float*)calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err / n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n) {
    int *x_gpu;
    size_t size = sizeof(int) * n;
    hipError_t status = hipMalloc((void **) &x_gpu, size);
    check_error(status);
    if (x) {
        status = hipMemcpy(x_gpu, x, size, hipMemcpyHostToDevice);
        check_error(status);
    }
    if (!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu) {
    hipError_t status = hipFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n) {
    size_t size = sizeof(float) * n;
    hipError_t status = hipMemcpy(x_gpu, x, size, hipMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n) {
    size_t size = sizeof(float) * n;
    hipError_t status = hipMemcpy(x, x_gpu, size, hipMemcpyDeviceToHost);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n) {
    float *temp = (float*)calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}

#else

void cuda_set_device(int n) {}

#endif
