#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "blas.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .00001f);
}

__global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}

extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
}

__global__ void spatial_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *spatial_variance_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= batch*filters) return;
    int f = i%filters;
    int b = i/filters;

    int k;
    spatial_variance_delta[i] = 0;
    for (k = 0; k < spatial; ++k) {
        int index = b*filters*spatial + f*spatial + k;
        spatial_variance_delta[i] += delta[index]*(x[index] - mean[f]);
    }
    spatial_variance_delta[i] *= -.5 * pow(variance[f] + .00001f, (float)(-3./2.));
}

extern "C" void variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    variance_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void accumulate_kernel(float *x, int n, int groups, float *sum)
{
    int k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}

extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *spatial_variance_delta, float *variance_delta)
{
    spatial_variance_delta_kernel<<<cuda_gridsize(filters*batch), BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, spatial_variance_delta);
    check_error(cudaPeekAtLastError());
    accumulate_kernel<<<cuda_gridsize(filters), BLOCK>>>(spatial_variance_delta, batch, filters, variance_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void spatial_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *spatial_mean_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= batch*filters) return;
    int f = i%filters;
    int b = i/filters;

    int k;
    spatial_mean_delta[i] = 0;
    for (k = 0; k < spatial; ++k) {
        int index = b*filters*spatial + f*spatial + k;
        spatial_mean_delta[i] += delta[index];
    }
    spatial_mean_delta[i] *= (-1./sqrt(variance[f] + .00001f));
}

extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *spatial_mean_delta, float *mean_delta)
{
    spatial_mean_delta_kernel<<<cuda_gridsize(filters*batch), BLOCK>>>(delta, variance, batch, filters, spatial, spatial_mean_delta);
    check_error(cudaPeekAtLastError());
    accumulate_kernel<<<cuda_gridsize(filters), BLOCK>>>(spatial_mean_delta, batch, filters, mean_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
}

extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__global__ void spatial_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(spatial*batch-1);
    int k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= batch*filters) return;
    int f = i%filters;
    int b = i/filters;

    variance[i] = 0;
    for(k = 0; k < spatial; ++k){
        int index = b*filters*spatial + f*spatial + k;
        variance[i] += pow((x[index] - mean[f]), 2);
    }
    variance[i] *= scale;
}

__global__ void variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial);
    int j,k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += pow((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

__global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] *= ALPHA;
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void mask_kernel(int n,  float *x, float mask_num, float *mask)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] = mask_num;
}

__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}

extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *spatial_mean, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters*batch), BLOCK>>>(x, 1, filters*batch, spatial, spatial_mean);
    check_error(cudaPeekAtLastError());
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(spatial_mean, batch, filters, 1, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *spatial_variance, float *variance)
{
    spatial_variance_kernel<<<cuda_gridsize(batch*filters), BLOCK>>>(x, mean, batch, filters, spatial, spatial_variance);
    check_error(cudaPeekAtLastError());
    accumulate_kernel<<<cuda_gridsize(filters), BLOCK>>>(spatial_variance, batch, filters, variance);
    check_error(cudaPeekAtLastError());
}

extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}

extern "C" void axpy_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_ongpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

extern "C" void pow_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void axpy_ongpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_ongpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_ongpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

extern "C" void mul_ongpu(int N, float * X, int INCX, float * Y, int INCY)
{
    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_ongpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void mask_ongpu(int N, float * X, float mask_num, float * mask)
{
    mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask);
    check_error(cudaPeekAtLastError());
}

extern "C" void const_ongpu(int N, float ALPHA, float * X, int INCX)
{
    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void scal_ongpu(int N, float ALPHA, float * X, int INCX)
{
    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void fill_ongpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
