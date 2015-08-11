#ifndef CUDA_H
#define CUDA_H

extern int gpu_index;

#ifdef GPU

#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
float *cuda_make_array(float *x, int n);
int *cuda_make_int_array(int n);
void cuda_push_array(float *x_gpu, float *x, int n);
void cuda_pull_array(float *x_gpu, float *x, int n);
void cuda_free(float *x_gpu);
void cuda_random(float *x_gpu, int n);
float cuda_compare(float *x_gpu, float *x, int n, char *s);
dim3 cuda_gridsize(size_t n);

#endif
#endif
