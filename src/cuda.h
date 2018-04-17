#ifndef CUDA_H
#define CUDA_H

#if defined(_MSC_VER) && _MSC_VER < 1900
	#define inline __inline
#endif

#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API
#else
#define YOLODLL_API
#endif
#endif

extern int gpu_index;

#ifdef GPU

#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#include "cudnn.h"
#endif // CUDNN

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
	void check_error(cudaError_t status);
	cublasHandle_t blas_handle();
	float *cuda_make_array(float *x, size_t n);
	int *cuda_make_int_array(size_t n);
	void cuda_push_array(float *x_gpu, float *x, size_t n);
	void cuda_pull_array(float *x_gpu, float *x, size_t n);
	YOLODLL_API void cuda_set_device(int n);
	int cuda_get_device();
	void cuda_free(float *x_gpu);
	void cuda_random(float *x_gpu, size_t n);
	float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
	dim3 cuda_gridsize(size_t n);
	cudaStream_t get_cuda_stream();
#ifdef __cplusplus
}
#endif // __cplusplus

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
enum {cudnn_fastest, cudnn_smallest};
#endif

#else // GPU
YOLODLL_API void cuda_set_device(int n);
#endif // GPU
#endif // CUDA_H
