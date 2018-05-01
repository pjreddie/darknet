#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU


#ifdef __cplusplus 
extern "C" {
#endif
void check_error(cudaError_t status);
void check_cublas_error(cublasStatus_t status);

cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
void cuda_dump_mem_stat();

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
void cudnn_handle_reset();
void blas_handle_reset();
#endif

#ifdef __cplusplus 
}
#endif


#endif
#endif
