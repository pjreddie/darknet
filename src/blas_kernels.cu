#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <assert.h>

#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"
#include "tree.h"

__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock, 0, get_cuda_stream()>>>(output, biases, n, size);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    backward_scale_kernel<<<n, BLOCK, 0, get_cuda_stream() >>>(x_norm, delta, batch, n, size, scale_updates);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void add_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] += biases[filter];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    add_bias_kernel<<<dimGrid, dimBlock, 0, get_cuda_stream()>>>(output, biases, n, size);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

/*
__global__ void dot_kernel(float *output, float scale, int batch, int n, int size, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int f1 = index / n;
    int f2 = index % n;
    if (f2 <= f1) return;

    float sum = 0;
    float norm1 = 0;
    float norm2 = 0;
    int b, i;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            sum += output[i1] * output[i2];
            norm1 += output[i1] * output[i1];
            norm2 += output[i2] * output[i2];
        }
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    float norm = norm1 * norm2;
    sum = sum / norm;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            delta[i1] += - scale * sum * output[i2] / norm;
            delta[i2] += - scale * sum * output[i1] / norm;
        }
    }
}

void dot_error_gpu(layer l)
{
    dot_kernel<<<cuda_gridsize(l.n*l.n), BLOCK, 0, get_cuda_stream()>>>(l.output_gpu, l.dot, l.batch, l.n, l.out_w * l.out_h, l.delta_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}
*/

void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    backward_bias_kernel<<<n, BLOCK, 0, get_cuda_stream() >>>(bias_updates, delta, batch, n, size);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void adam_kernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;

    float mhat = m[index] / (1.f - powf(B1, t));
    float vhat = v[index] / (1.f - powf(B2, t));

    x[index] = x[index] + rate * mhat / (sqrtf(vhat) + eps);
}

extern "C" void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    adam_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(n, x, m, v, B1, B2, rate, eps, t);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_ongpu(n, B1, m, 1);
    scal_ongpu(n, B2, v, 1);
    axpy_ongpu(n, -decay*batch, w, 1, d, 1);

    axpy_ongpu(n, (1 - B1), d, 1, m, 1);
    mul_ongpu(n, d, 1, d, 1);
    axpy_ongpu(n, (1 - B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
    fill_ongpu(n, 0, d, 1);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    x[index] = (x[index] - mean[f])/(sqrtf(variance[f]) + .000001f);
}

__global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    delta[index] = delta[index] * 1.F/(sqrtf(variance[f]) + .000001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}

extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
    CHECK_CUDA(cudaPeekAtLastError());
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
    variance_delta[i] *= -.5 * powf(variance[i] + .000001f, (float)(-3./2.));
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

__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.F/sqrtf(variance[filter] + .000001f));
    }
}

__global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5 * powf(variance[filter] + .000001f, (float)(-3./2.));
    }
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
    mean_delta[i] *= (-1.F/sqrtf(variance[i] + .000001f));
}

extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK, 0, get_cuda_stream() >>>(delta, variance, batch, filters, spatial, mean_delta);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    fast_mean_delta_kernel<<<filters, BLOCK, 0, get_cuda_stream() >>>(delta, variance, batch, filters, spatial, mean_delta);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    fast_variance_delta_kernel<<<filters, BLOCK, 0, get_cuda_stream() >>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1.F/(batch * spatial);
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

__global__ void variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1.F/(batch * spatial - 1);
    int j,k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += powf((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

   // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = powf(X[i*INCX], ALPHA);
}

__global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void constrain_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = fminf(ALPHA, fmaxf(-ALPHA, X[i*INCX]));
}

__global__ void supp_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}

__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] *= ALPHA;
}

__global__ void scal_add_kernel(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void mask_kernel_new_api(int n, float *x, float mask_num, float *mask, float val)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n && mask[i] == mask_num) x[i] = val;
}

__global__ void mask_kernel(int n, float *x, float mask_num, float *mask)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] = mask_num;
}

__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

__global__ void simple_copy_kernel(int size, float *src, float *dst)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size)
        dst[index] = src[index];
}

__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}


extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream()>>>(N, x, mean, variance, batch, filters, spatial);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }
    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel<<<filters, BLOCK, 0, get_cuda_stream()>>>(x, batch, filters, spatial, mean);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BLOCK, 0, get_cuda_stream() >>>(x, mean, batch, filters, spatial, variance);
    CHECK_CUDA(cudaPeekAtLastError());
}


extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK, 0, get_cuda_stream() >>>(x, batch, filters, spatial, mean);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK, 0, get_cuda_stream() >>>(x, mean, batch, filters, spatial, variance);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void axpy_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_ongpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

extern "C" void pow_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    pow_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, ALPHA, X, INCX, Y, INCY);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void axpy_ongpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream()>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void copy_ongpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_ongpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

extern "C" void simple_copy_ongpu(int size, float *src, float *dst)
{
    const int num_blocks = size / BLOCK + 1;
    simple_copy_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(size, src, dst);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void mul_ongpu(int N, float * X, int INCX, float * Y, int INCY)
{
    mul_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, X, INCX, Y, INCY);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void copy_ongpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream()>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void flatten_kernel(int N, float *x, int spatial, int layers, int batch, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward) out[i2] = x[i1];
    else out[i1] = x[i2];
}

extern "C" void flatten_ongpu(float *x, int spatial, int layers, int batch, int forward, float *out)
{
    int size = spatial*batch*layers;
    flatten_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream()>>>(size, x, spatial, layers, batch, forward, out);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int size = w*h*c*batch;
    reorg_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream()>>>(size, x, w, h, c, batch, stride, forward, out);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void mask_gpu_new_api(int N, float * X, float mask_num, float * mask, float val)
{
	mask_kernel_new_api <<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, X, mask_num, mask, val);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void mask_ongpu(int N, float * X, float mask_num, float * mask)
{
    mask_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, X, mask_num, mask);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void const_ongpu(int N, float ALPHA, float * X, int INCX)
{
    const_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, ALPHA, X, INCX);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void constrain_ongpu(int N, float ALPHA, float * X, int INCX)
{
    constrain_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, ALPHA, X, INCX);
    CHECK_CUDA(cudaPeekAtLastError());
}


extern "C" void scal_ongpu(int N, float ALPHA, float * X, int INCX)
{
    scal_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream()>>>(N, ALPHA, X, INCX);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void scal_add_ongpu(int N, float ALPHA, float BETA, float * X, int INCX)
{
    scal_add_kernel << <cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >> >(N, ALPHA, BETA, X, INCX);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void supp_ongpu(int N, float ALPHA, float * X, int INCX)
{
    supp_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >>>(N, ALPHA, X, INCX);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void fill_ongpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream()>>>(N, ALPHA, X, INCX);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] += add[add_index];
}

extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    shortcut_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream()>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void simple_input_shortcut_kernel(float *in, int size, float *add, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    out[id] = in[id] + add[id];
}

__global__ void input_shortcut_kernel(float *in, int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = in[out_index] + add[add_index];
}

extern "C" void input_shortcut_gpu(float *in, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    if (w1 == w2 && h1 == h2 && c1 == c2) {
        int size = batch * w1 * h1 * c1;
        simple_input_shortcut_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> >(in, size, add, out);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    //input_shortcut_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> >(in, size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);
    simple_copy_ongpu(w2 * h2 * c2 * batch, in, out);
    shortcut_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> >(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void smooth_l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        float abs_val = abs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? -1 : 1;
        }
    }
}

extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    smooth_l1_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(n, pred, truth, delta, error);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		float t = truth[i];
		float p = pred[i];
		error[i] = (t) ? -log(p) : 0;
		delta[i] = t - p;
	}
}

extern "C" void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
	softmax_x_ent_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(n, pred, truth, delta, error);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void l2_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff; //I know this is technically wrong, deal with it.
        delta[i] = diff;
    }
}

extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    l2_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(n, pred, truth, delta, error);
    CHECK_CUDA(cudaPeekAtLastError());
}



__global__ void weighted_sum_kernel(int n, float *a, float *b, float *s, float *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
{
    weighted_sum_kernel<<<cuda_gridsize(num), BLOCK, 0, get_cuda_stream() >>>(num, a, b, s, c);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void weighted_delta_kernel(int n, float *a, float *b, float *s, float *da, float *db, float *ds, float *dc)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * a[i] + dc[i] * -b[i];
    }
}

extern "C" void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc)
{
    weighted_delta_kernel<<<cuda_gridsize(num), BLOCK, 0, get_cuda_stream() >>>(num, a, b, s, da, db, ds, dc);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void mult_add_into_kernel(int n, float *a, float *b, float *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] += a[i]*b[i];
    }
}

extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
{
    mult_add_into_kernel<<<cuda_gridsize(num), BLOCK, 0, get_cuda_stream() >>>(num, a, b, c);
    CHECK_CUDA(cudaPeekAtLastError());
}


__device__ void softmax_device(int n, float *input, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}

__global__ void softmax_kernel(int n, int offset, int batch, float *input, float temp, float *output)
{
    int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(b >= batch) return;
    softmax_device(n, input + b*offset, temp, output + b*offset);
}

extern "C" void softmax_gpu(float *input, int n, int offset, int groups, float temp, float *output)
{
    int inputs = n;
    int batch = groups;
    softmax_kernel<<<cuda_gridsize(batch), BLOCK, 0, get_cuda_stream()>>>(inputs, offset, batch, input, temp, output);
    CHECK_CUDA(cudaPeekAtLastError());
}

__device__ void softmax_device_new_api(float *input, int n, float temp, int stride, float *output)
{
	int i;
	float sum = 0;
	float largest = -INFINITY;
	for (i = 0; i < n; ++i) {
		int val = input[i*stride];
		largest = (val>largest) ? val : largest;
	}
	for (i = 0; i < n; ++i) {
		float e = expf(input[i*stride] / temp - largest / temp);
		sum += e;
		output[i*stride] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i*stride] /= sum;
	}
}

__global__ void softmax_kernel_new_api(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= batch*groups) return;
	int b = id / groups;
	int g = id % groups;
	softmax_device_new_api(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

extern "C" void softmax_gpu_new_api(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	softmax_kernel_new_api << <cuda_gridsize(batch*groups), BLOCK, 0, get_cuda_stream() >> >(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int out_index = i;
    int out_w = i % (w*stride);
    i = i / (w*stride);
    int out_h = i % (h*stride);
    i = i / (h*stride);
    int out_c = i%c;
    i = i / c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if (forward) out[out_index] += scale * x[in_index];
    else atomicAdd(x + in_index, scale * out[out_index]);
}

extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> >(size, in, w, h, c, batch, stride, forward, scale, out);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void softmax_tree_kernel(float *input, int spatial, int batch, int stride, float temp, float *output, int groups, int *group_size, int *group_offset)
{
	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= spatial*batch*groups) return;
	int s = id % spatial;
	id = id / spatial;
	int g = id % groups;
	int b = id / groups;
	int goff = group_offset[g] * spatial;
	int boff = b*stride;
	softmax_device_new_api(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}

extern "C" void softmax_tree_gpu(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier)
{
	int *tree_groups_size = cuda_make_int_array_new_api(hier.group_size, hier.groups);
	int *tree_groups_offset = cuda_make_int_array_new_api(hier.group_offset, hier.groups);
	/*
	static int *tree_groups_size = 0;
	static int *tree_groups_offset = 0;
	if(!tree_groups_size){
	tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
	tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
	}
	*/
	int num = spatial*batch*hier.groups;
	softmax_tree_kernel <<<cuda_gridsize(num), BLOCK, 0, get_cuda_stream() >>>(input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
    CHECK_CUDA(cudaPeekAtLastError());
	cuda_free((float *)tree_groups_size);
	cuda_free((float *)tree_groups_offset);
}


__global__ void fix_nan_and_inf_kernel(float *input, size_t size)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        float val = input[index];
        if (isnan(val) || isinf(val))
            input[index] = 1.0f / index;  // pseudo random value
    }
}

extern "C" void fix_nan_and_inf(float *input, size_t size)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    fix_nan_and_inf_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(input, size);
    CHECK_CUDA(cudaPeekAtLastError());
    //CHECK_CUDA(cudaDeviceSynchronize());
}


__global__ void is_nan_or_inf_kernel(float *input, size_t size, int *pinned_return)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        float val = input[index];
        if (isnan(val) || isinf(val))
            *pinned_return = 1;
    }
}

extern "C" int is_nan_or_inf(float *input, size_t size)
{
    int *pinned_return;
    CHECK_CUDA(cudaHostAlloc(&pinned_return, sizeof(int), cudaHostRegisterMapped));
    *pinned_return = 0;

    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    is_nan_or_inf_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(input, size, pinned_return);
    CHECK_CUDA(cudaDeviceSynchronize());
    int ret_val = *pinned_return;

    CHECK_CUDA(cudaFreeHost(pinned_return));
    return ret_val;
}

__global__ void add_3_arrays_activate_kernel(float *a1, float *a2, float *a3, size_t size, ACTIVATION a, float *dst)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        float val = 0;
        val += a1[index];
        val += a2[index];
        if (a3) val += a3[index];
        if (a == LOGISTIC) val = 1.f / (1.f + expf(-val));
        else if(a == TANH) val = (2 / (1 + expf(-2 * val)) - 1);
        dst[index] = val;
    }
}

extern "C" void add_3_arrays_activate(float *a1, float *a2, float *a3, size_t size, ACTIVATION a, float *dst)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    if (a != LOGISTIC && a != TANH) {
        printf(" add_3_arrays_activate() doesn't support activation %d, it supports only LOGISTIC and TANH \n", a);
        exit(EXIT_FAILURE);
    }
    add_3_arrays_activate_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(a1, a2, a3, size, a, dst);
}


__global__ void sum_of_mults_kernel(float *a1, float *a2, float *b1, float *b2, size_t size, float *dst)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        dst[index] = a1[index] * a2[index] + b1[index] * b2[index];
    }
}

extern "C" void sum_of_mults(float *a1, float *a2, float *b1, float *b2,  size_t size, float *dst)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    sum_of_mults_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(a1, a2, b1, b2, size, dst);
}


__global__ void activate_and_mult_kernel(float *a1, float *a2, size_t size, ACTIVATION a, float *dst)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        float val = a1[index];
        if (a == TANH) val = (2 / (1 + expf(-2 * val)) - 1);
        dst[index] = val * a2[index];
    }
}

extern "C" void activate_and_mult(float *a1, float *a2, size_t size, ACTIVATION a, float *dst)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    if (a != TANH) {
        printf(" activat_and_mult() doesn't support activation %d, it supports only TANH \n", a);
        exit(EXIT_FAILURE);
    }
    activate_and_mult_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(a1, a2, size, a, dst);
}
