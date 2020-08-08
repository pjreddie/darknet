#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <assert.h>
#include <float.h>

#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"
#include "tree.h"

__inline__ __device__
float warpAllReduceSum(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
#if CUDART_VERSION >= 9000
        val += __shfl_xor_sync(0xffffffff, val, mask);
#else
        val += __shfl_xor(val, mask);
#endif
    return val;
}

__global__ void compare_2_arrays_kernel(float *one, float *two, int size)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= size) return;

    const float diff = 100 * fabs(one[index] - two[index]) / fabs(one[index]);

    if (diff > 10) printf(" i: %d - one = %f, two = %f, diff = %f %% \n", index, one[index], two[index], diff);
}

void compare_2_arrays_gpu(float *one, float *two, int size)
{
    const int num_blocks = get_number_of_blocks(size, BLOCK);

    compare_2_arrays_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(one, two, size);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void mean_array_kernel(float *src, int size, float alpha, float *avg)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= size) return;

    avg[i] = avg[i] * (1 - alpha) + src[i] * alpha;
    src[i] = avg[i];
}


void mean_array_gpu(float *src, int size, float alpha, float *avg)
{
    const int num_blocks = get_number_of_blocks(size, BLOCK);

    mean_array_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(src, size, alpha, avg);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void scale_bias_kernel(float *output, float *scale, int batch, int filters, int spatial, int current_size)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= current_size) return;

    int f = (index / spatial) % filters;
    output[index] *= scale[f];
}

void scale_bias_gpu(float *output, float *scale, int batch, int filters, int spatial)
{
    const int current_size = batch * filters * spatial;
    const int num_blocks = get_number_of_blocks(current_size, BLOCK);

    scale_bias_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(output, scale, batch, filters, spatial, current_size);
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

__global__ void add_bias_kernel(float *output, float *biases, int batch, int filters, int spatial, int current_size)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= current_size) return;

    int f = (index / spatial) % filters;
    output[index] += biases[f];
}

void add_bias_gpu(float *output, float *biases, int batch, int filters, int spatial)
{
    const int current_size = batch * filters * spatial;
    const int num_blocks = get_number_of_blocks(current_size, BLOCK);

    add_bias_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(output, biases, batch, filters, spatial, current_size);
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
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index / spatial) % filters;

    x[index] = (x[index] - mean[f]) / (sqrtf(variance[f] + .00001f));
}

extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    const int current_size = batch * filters * spatial;
    const int num_blocks = get_number_of_blocks(current_size, BLOCK);

    normalize_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(current_size, x, mean, variance, batch, filters, spatial);
    CHECK_CUDA(cudaPeekAtLastError());
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

__global__ void constrain_weight_updates_kernel(int N, float coef, float *weights_gpu, float *weight_updates_gpu)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        const float w = weights_gpu[i];
        const float wu = weight_updates_gpu[i];
        const float wu_sign = (wu == 0) ? 0 : (fabs(wu) / wu);
        const float abs_limit = fabs(w * coef);
        if (fabs(wu) > abs_limit) weight_updates_gpu[i] = abs_limit * wu_sign;
    }
}

extern "C" void constrain_weight_updates_ongpu(int N, float coef, float *weights_gpu, float *weight_updates_gpu)
{
    constrain_weight_updates_kernel << <cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >> >(N, coef, weights_gpu, weight_updates_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
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
__global__ void constrain_min_max_kernel(int N, float MIN, float MAX, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) X[i*INCX] = fminf(MAX, fmaxf(MIN, X[i*INCX]));
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
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= N) return;
    X[index*INCX] = ALPHA;
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
        float mean_tmp = 0;
        for(i = 0; i < threads; ++i){
            mean_tmp += local[i];
        }
        mean_tmp /= spatial * batch;
        mean[filter] = mean_tmp;
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel << <filters, BLOCK, 0, get_cuda_stream() >> >(x, batch, filters, spatial, mean);
    CHECK_CUDA(cudaPeekAtLastError());
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
        float variance_tmp = 0;
        for(i = 0; i < threads; ++i){
            variance_tmp += local[i];
        }
        variance_tmp /= (spatial * batch);// -1);
        variance[filter] = variance_tmp;
    }
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BLOCK, 0, get_cuda_stream() >>>(x, mean, batch, filters, spatial, variance);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void  fast_v_cbn_kernel(const float *x, float *mean, int batch, int filters, int spatial, int minibatch_index, int max_minibatch_index, float *m_avg, float *v_avg, float *variance,
    const float alpha, float *rolling_mean_gpu, float *rolling_variance_gpu, int inverse_variance, float epsilon)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for (j = 0; j < batch; ++j) {
        for (i = 0; i < spatial; i += threads) {
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i + id < spatial) ? powf(x[index], 2) : 0;
        }
    }
    __syncthreads();

    if (id == 0) {
        float v_tmp = 0;
        v_tmp = 0;
        for (i = 0; i < threads; ++i) {
            v_tmp += local[i];
        }
        v_tmp /= (spatial * batch - 1);

        v_tmp = fmax(v_tmp, powf(mean[filter], 2));


        const float alpha_cbn = 1.0f / minibatch_index;

        m_avg[filter] = alpha_cbn * mean[filter] + (1 - alpha_cbn) * m_avg[filter];
        mean[filter] = m_avg[filter];

        v_avg[filter] = alpha_cbn * v_tmp + (1 - alpha_cbn) * v_avg[filter];

        float variance_tmp = fmax(0.0f, v_avg[filter] - powf(m_avg[filter], 2));
        if (inverse_variance) variance[filter] = 1.0f / sqrtf(variance_tmp + epsilon);
        else variance[filter] = variance_tmp;

        //if (max_minibatch_index == minibatch_index)
        {
            if(rolling_mean_gpu) rolling_mean_gpu[filter] = alpha * mean[filter] + (1 - alpha) * rolling_mean_gpu[filter];

            if(rolling_variance_gpu) rolling_variance_gpu[filter] = alpha * variance_tmp + (1 - alpha) * rolling_variance_gpu[filter];
        }
    }
}

extern "C" void fast_v_cbn_gpu(const float *x, float *mean, int batch, int filters, int spatial, int minibatch_index, int max_minibatch_index, float *m_avg, float *v_avg, float *variance,
    const float alpha, float *rolling_mean_gpu, float *rolling_variance_gpu, int inverse_variance, float epsilon)
{
    fast_v_cbn_kernel << <filters, BLOCK, 0, get_cuda_stream() >> >(x, mean, batch, filters, spatial, minibatch_index, max_minibatch_index, m_avg, v_avg, variance, alpha, rolling_mean_gpu, rolling_variance_gpu, inverse_variance, epsilon);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void inverse_variance_kernel(int size, float *src, float *dst, float epsilon)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size)
        dst[index] = 1.0f / sqrtf(src[index] + epsilon);
}

extern "C" void inverse_variance_ongpu(int size, float *src, float *dst, float epsilon)
{
    const int num_blocks = size / BLOCK + 1;
    inverse_variance_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(size, src, dst, epsilon);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void normalize_scale_bias_kernel(int N, float *x, float *mean, float *variance, float *scales, float *biases, int batch, int filters, int spatial, int inverse_variance, float epsilon)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index / spatial) % filters;

    float val = 0;
    if(inverse_variance) val = (x[index] - mean[f]) * variance[f];
    else val = (x[index] - mean[f]) / (sqrtf(variance[f] + epsilon));
    val *= scales[f];
    val += biases[f];

    if (!isnan(val) && !isinf(val))
        x[index] = val;
}

extern "C" void normalize_scale_bias_gpu(float *x, float *mean, float *variance, float *scales, float *biases, int batch, int filters, int spatial, int inverse_variance, float epsilon)
{
    const int current_size = batch * filters * spatial;
    const int num_blocks = get_number_of_blocks(current_size, BLOCK);

    normalize_scale_bias_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(current_size, x, mean, variance, scales, biases, batch, filters, spatial, inverse_variance, epsilon);
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

extern "C" void memcpy_ongpu(void *dst, void *src, int size_bytes)
{
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyDefault, get_cuda_stream()));
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

extern "C" void constrain_min_max_ongpu(int N, float MIN, float MAX, float * X, int INCX)
{
    constrain_min_max_kernel << <cuda_gridsize(N), BLOCK, 0, get_cuda_stream() >> >(N, MIN, MAX, X, INCX);
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
    //fill_kernel<<<cuda_gridsize(N), BLOCK, 0, get_cuda_stream()>>>(N, ALPHA, X, INCX);
    //CHECK_CUDA(cudaPeekAtLastError());
    fill_kernel << <get_number_of_blocks(N, BLOCK), BLOCK, 0, get_cuda_stream() >> >(N, ALPHA, X, INCX);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void gradient_centralization_kernel(int filters, int f_size, float *in)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = index % WARP_SIZE;
    const int f = index / WARP_SIZE;

    if (f >= filters) return;

    float mean = 0;
    for (int i = 0; i < f_size; i += WARP_SIZE) {
        mean += warpAllReduceSum(in[f*f_size + i + tid]);
    }
    mean = mean / f_size;
    for (int i = 0; i < f_size; i += WARP_SIZE) {
        in[f*f_size + i + tid] -= mean;
    }

}

extern "C" void gradient_centralization_gpu(int w, int h, int c, int f, float *in)
{
    const int size = f * WARP_SIZE;
    const int f_size = c * h * w;
    if (f_size % WARP_SIZE == 0) {

        gradient_centralization_kernel << <get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >> > (f, f_size, in);
        CHECK_CUDA(cudaPeekAtLastError());
    }
}

__device__ float relu(float src) {
    if (src > 0) return src;
    return 0;
}

__device__ float lrelu(float src) {
    const float eps = 0.001;
    if (src > eps) return src;
    return eps;
}

__device__ float grad_relu(float src) {
    return (src > 0);
}

__device__ float grad_lrelu(float src) {
    const float eps = 0.001;
    return (src > eps);
}

__global__ void shortcut_singlelayer_simple_kernel(int size, int src_outputs, int batch, int n, int *outputs_of_layers_gpu, float **layers_output_gpu, float *out, float *in, float *weights_gpu, int nweights, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    const int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    int src_id = id;
    const int src_i = src_id % src_outputs;
    src_id /= src_outputs;
    int src_b = src_id;

    float out_val = in[id];

    int add_outputs = outputs_of_layers_gpu[0];
    if (src_i < add_outputs) {
        int add_index = add_outputs*src_b + src_i;

        float *add = layers_output_gpu[0];
        out_val += add[add_index];
    }
    out[id] = out_val;
}

__global__ void shortcut_multilayer_kernel(int size, int src_outputs, int batch, int n, int *outputs_of_layers_gpu, float **layers_output_gpu, float *out, float *in, float *weights_gpu, int nweights, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    const int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int src_id = id;
    const int src_i = src_id % src_outputs;
    src_id /= src_outputs;
    int src_b = src_id;

    float sum = 1, max_val = -FLT_MAX;
    if (weights_gpu && weights_normalization) {
        if (weights_normalization == SOFTMAX_NORMALIZATION) {
            for (int i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights_gpu[weights_index];
                if (max_val < w) max_val = w;
            }
        }
        const float eps = 0.0001;
        sum = eps;
        for (int i = 0; i < (n + 1); ++i) {
            const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
            const float w = weights_gpu[weights_index];
            if (weights_normalization == RELU_NORMALIZATION) sum += lrelu(w);
            else if (weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
        }
    }

    float out_val = 0;

    if (weights_gpu) {
        float w = weights_gpu[src_i / step];
        if (weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
        else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

        out_val = in[id] * w; // [0 or c or (c, h ,w)]
    }
    else out_val = in[id];

    // layers
    for (int i = 0; i < n; ++i) {
        int add_outputs = outputs_of_layers_gpu[i];
        if (src_i < add_outputs) {
            int add_index = add_outputs*src_b + src_i;

            float *add = layers_output_gpu[i];

            if (weights_gpu) {
                const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                float w = weights_gpu[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
                else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                out_val += add[add_index] * w; // [0 or c or (c, h ,w)]
            }
            else out_val += add[add_index];
        }
    }
    out[id] = out_val;
}

extern "C" void shortcut_multilayer_gpu(int src_outputs, int batch, int n, int *outputs_of_layers_gpu, float **layers_output_gpu, float *out, float *in, float *weights_gpu, int nweights, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    //printf(" src_outputs = %d, batch = %d, n = %d \n", src_outputs, batch, n);
    int size = batch * src_outputs;
    if (nweights == 0 && n == 1) {
        shortcut_singlelayer_simple_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (size, src_outputs, batch, n, outputs_of_layers_gpu, layers_output_gpu, out, in, weights_gpu, nweights, weights_normalization);
    }
    else {
        shortcut_multilayer_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (size, src_outputs, batch, n, outputs_of_layers_gpu, layers_output_gpu, out, in, weights_gpu, nweights, weights_normalization);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void backward_shortcut_multilayer_kernel(int size, int src_outputs, int batch, int n, int *outputs_of_layers_gpu,
    float **layers_delta_gpu, float *delta_out, float *delta_in, float *weights_gpu, float *weight_updates_gpu, int nweights, float *in, float **layers_output_gpu, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    const int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int src_id = id;
    const int src_i = src_id % src_outputs;
    src_id /= src_outputs;
    int src_b = src_id;

    float grad = 1, sum = 1, max_val = -FLT_MAX;
    int i;
    if (weights_gpu && weights_normalization) {
        if (weights_normalization == SOFTMAX_NORMALIZATION) {
            for (int i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                float w = weights_gpu[weights_index];
                if (max_val < w) max_val = w;
            }
        }
        const float eps = 0.0001;
        sum = eps;
        for (i = 0; i < (n + 1); ++i) {
            const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
            const float w = weights_gpu[weights_index];
            if (weights_normalization == RELU_NORMALIZATION) sum += lrelu(w);
            else if (weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
        }

    }

    if (weights_gpu) {
        float w = weights_gpu[src_i / step];
        if (weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
        else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

        if (weights_normalization == RELU_NORMALIZATION) grad = w;
        else if (weights_normalization == SOFTMAX_NORMALIZATION) grad = w*(1-w);

        delta_out[id] += delta_in[id] * w; // [0 or c or (c, h ,w)]
        float weights_update_tmp = delta_in[id] * in[id] * grad;// / step;

        if (layer_step == 1 && (size/32) > (id/32 + 1)) {
            if (isnan(weights_update_tmp) || isinf(weights_update_tmp)) {
                weights_update_tmp = 0;
            }
            float wu = warpAllReduceSum(weights_update_tmp);
            if (threadIdx.x % 32 == 0) {
                if (!isnan(wu) && !isinf(wu))
                    atomicAdd(&weight_updates_gpu[src_i / step], wu);
            }
        }
        else {
            if (!isnan(weights_update_tmp) && !isinf(weights_update_tmp))
                atomicAdd(&weight_updates_gpu[src_i / step], weights_update_tmp);
                //weight_updates_gpu[src_i / step] += weights_update_tmp;
        }
    }
    else delta_out[id] += delta_in[id];

    // layers
    for (int i = 0; i < n; ++i) {
        int add_outputs = outputs_of_layers_gpu[i];
        if (src_i < add_outputs) {
            int add_index = add_outputs*src_b + src_i;
            int out_index = id;

            float *layer_delta = layers_delta_gpu[i];
            if (weights_gpu) {
                float *add = layers_output_gpu[i];

                const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                float w = weights_gpu[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
                else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                if (weights_normalization == RELU_NORMALIZATION) grad = w;
                else if (weights_normalization == SOFTMAX_NORMALIZATION) grad = w*(1 - w);

                layer_delta[add_index] += delta_in[id] * w;
                float weights_update_tmp = delta_in[id] * add[add_index] * grad;// / step;

                if (layer_step == 1 && (size / 32) > (id / 32 + 1)) {
                    if (isnan(weights_update_tmp) || isinf(weights_update_tmp)) {
                        weights_update_tmp = 0;
                    }
                    float wu = warpAllReduceSum(weights_update_tmp);
                    if (threadIdx.x % 32 == 0) {
                        if (!isnan(wu) && !isinf(wu))
                            atomicAdd(&weight_updates_gpu[weights_index], wu);
                        //if(weights_gpu[weights_index] != 1) printf(" wu = %f, weights_update_tmp = %f, w = %f, weights_gpu[weights_index] = %f, grad = %f, weights_normalization = %d ",
                        //    wu, weights_update_tmp, w, weights_gpu[weights_index], grad, weights_normalization);
                    }
                }
                else {
                    if (!isnan(weights_update_tmp) && !isinf(weights_update_tmp))
                        atomicAdd(&weight_updates_gpu[weights_index], weights_update_tmp);
                        //weight_updates_gpu[weights_index] += weights_update_tmp;
                }
            }
            else layer_delta[add_index] += delta_in[id];
        }
    }
}

extern "C" void backward_shortcut_multilayer_gpu(int src_outputs, int batch, int n, int *outputs_of_layers_gpu,
    float **layers_delta_gpu, float *delta_out, float *delta_in, float *weights_gpu, float *weight_updates_gpu, int nweights, float *in, float **layers_output_gpu, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1
    //printf(" nweights = %d, n = %d, layer_step = %d, step = %d \n", nweights, n, layer_step, step);

    //printf(" src_outputs = %d, batch = %d, n = %d \n", src_outputs, batch, n);
    int size = batch * src_outputs;
    backward_shortcut_multilayer_kernel << <cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >> > (size, src_outputs, batch, n, outputs_of_layers_gpu,
        layers_delta_gpu, delta_out, delta_in, weights_gpu, weight_updates_gpu, nweights, in, layers_output_gpu, weights_normalization);
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
        if (isnan(val) || isinf(val)) {
            input[index] = 1.0f / (fabs((float)index) + 1);  // pseudo random value
        }
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


__global__ void reset_nan_and_inf_kernel(float *input, size_t size)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        float val = input[index];
        if (isnan(val) || isinf(val)) {
            input[index] = 0;
        }
    }
}

extern "C" void reset_nan_and_inf(float *input, size_t size)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    reset_nan_and_inf_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(input, size);
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
        if (a1) val += a1[index];
        if (a2) val += a2[index];
        if (a3) val += a3[index];
        if (a == LOGISTIC) val = 1.f / (1.f + expf(-val));
        else if (a == TANH) val = (2 / (1 + expf(-2 * val)) - 1);
        else if (a == LEAKY) val = (val < 0) ? val*0.1 : val;
        dst[index] = val;
    }
}

extern "C" void add_3_arrays_activate(float *a1, float *a2, float *a3, size_t size, ACTIVATION a, float *dst)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    if (!(a == LOGISTIC || a == TANH || a == LEAKY || a == LINEAR)) {
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
        else if (a == LEAKY) val = (val < 0) ? val*0.1 : val;
        dst[index] = val * a2[index];
    }
}

extern "C" void activate_and_mult(float *a1, float *a2, size_t size, ACTIVATION a, float *dst)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    if (!(a == TANH || a == LEAKY || a == LINEAR)) {
        printf(" activat_and_mult() doesn't support activation %d, it supports only TANH \n", a);
        exit(EXIT_FAILURE);
    }
    activate_and_mult_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(a1, a2, size, a, dst);
}



__global__ void scale_channels_kernel(float *in_w_h_c, int size, int channel_size, int batch_size, int scale_wh, float *scales_c, float *out)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        if (scale_wh) {
            int osd_index = index % channel_size + (index / batch_size)*channel_size;

            out[index] = in_w_h_c[index] * scales_c[osd_index];
        }
        else {
            out[index] = in_w_h_c[index] * scales_c[index / channel_size];
        }
    }
}

extern "C" void scale_channels_gpu(float *in_w_h_c, int size, int channel_size, int batch_size, int scale_wh, float *scales_c, float *out)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    scale_channels_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(in_w_h_c, size, channel_size, batch_size, scale_wh, scales_c, out);
    CHECK_CUDA(cudaPeekAtLastError());
}




__global__ void backward_scale_channels_kernel(float *in_w_h_c_delta, int size, int channel_size, int batch_size, int scale_wh,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < size) {

        if (scale_wh)
        {
            int osd_index = index % channel_size + (index / batch_size)*channel_size;

            //out_state_delta[osd_index] += in_w_h_c_delta[index] * in_from_output[index]; // l.delta * from  (should be divided by channel_size?)
            atomicAdd(&out_state_delta[osd_index], in_w_h_c_delta[index] * in_from_output[index] / channel_size); // l.delta * from

            out_from_delta[index] += in_scales_c[osd_index] * in_w_h_c_delta[index]; // input * l.delta  // atomic isn't required here

        }
        else {
            int osd_index = index / channel_size;
            //out_state_delta[osd_index] += in_w_h_c_delta[index] * in_from_output[index]; // l.delta * from  (should be divided by channel_size?)

            int warp_id = index / 32;
            int index_warp_start = warp_id * 32;
            int osd_index_warp_start = index_warp_start / channel_size;
            int osd_index_warp_end = (index_warp_start + 31) / channel_size;

            if (osd_index_warp_start == osd_index_warp_end) // all thread in warp process the same channel
            {
                float sum = warpAllReduceSum(in_w_h_c_delta[index] * in_from_output[index]); // l.delta * from
                if (threadIdx.x % 32 == 0) {
                    atomicAdd(&out_state_delta[osd_index], sum);
                    //out_state_delta[osd_index] += sum;
                }
            }
            else {
                atomicAdd(&out_state_delta[osd_index], in_w_h_c_delta[index] * in_from_output[index]); // l.delta * from
            }

            out_from_delta[index] += in_scales_c[osd_index] * in_w_h_c_delta[index]; // input * l.delta  // atomic isn't required here
        }
    }
}

extern "C" void backward_scale_channels_gpu(float *in_w_h_c_delta, int size, int channel_size, int batch_size, int scale_wh,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    backward_scale_channels_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (in_w_h_c_delta, size, channel_size, batch_size, scale_wh,
        in_scales_c, out_from_delta,
        in_from_output, out_state_delta);

    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void sam_kernel(float *in_w_h_c, int size, int channel_size, float *scales_c, float *out)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = in_w_h_c[index] * scales_c[index];
    }
}

extern "C" void sam_gpu(float *in_w_h_c, int size, int channel_size, float *scales_c, float *out)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    sam_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(in_w_h_c, size, channel_size, scales_c, out);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void backward_sam_kernel(float *in_w_h_c_delta, int size, int channel_size,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        out_state_delta[index] += in_w_h_c_delta[index] * in_from_output[index]; // l.delta * from  (should be divided by channel_size?)
        out_from_delta[index] += in_scales_c[index] * in_w_h_c_delta[index]; // input * l.delta

                                                                             //out_state_delta[index] += in_w_h_c_delta[index];
                                                                             //out_from_delta[index] = in_w_h_c_delta[index];
    }
}

extern "C" void backward_sam_gpu(float *in_w_h_c_delta, int size, int channel_size,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    backward_sam_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (in_w_h_c_delta, size, channel_size,
        in_scales_c, out_from_delta,
        in_from_output, out_state_delta);

    CHECK_CUDA(cudaPeekAtLastError());
}


__global__  void smooth_rotate_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int kernel_size, int angle, int reverse)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int kernel_area = kernel_size * kernel_size;
    const int i = index * kernel_area;

    const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
    const int stage_id = index / stage_step;

    // nweights = (c / groups) * n * size * size;
    // kernel_area = size*size

    if (i < nweights)
    {
        // rotate left or right
        if (reverse) angle = -angle;

        const float cos_a = cosf(angle * 3.14159265 / 180);
        const float sin_a = sinf(angle * 3.14159265 / 180);
        const int x_c = kernel_size / 2;
        const int y_c = kernel_size / 2;

        float dropout_sum = 0;

        for (int y = 0; y < kernel_size; ++y) {
            for (int x = 0; x < kernel_size; ++x) {
                // Xsource = x*cos(alpha) + y*sin(alpha)
                // Ysource = -x*sin(alpha) + y*cos(alpha)

                float x_s = x_c + (x - x_c)*cos_a + (y - y_c)*sin_a;
                float y_s = y_c - (x - x_c)*sin_a + (y - y_c)*cos_a;

                int x_0 = floor(x_s);   // round down
                int x_1 = ceil(x_s);    // round up
                if (x_0 == x_1) x_1 = x_0 + 1;
                int y_0 = floor(y_s);
                int y_1 = ceil(y_s);
                if (y_0 == y_1) y_1 = y_0 + 1;

                float c_x_0 = x_1 - x_s;
                float c_x_1 = x_s - x_0;
                float c_y_0 = y_1 - y_s;
                float c_y_1 = y_s - y_0;


                float val = 0;
                if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                else dropout_sum += c_x_0 * c_y_0;

                if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                else dropout_sum += c_x_1 * c_y_0;

                if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                else dropout_sum += c_x_0 * c_y_1;

                if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                else dropout_sum += c_x_1 * c_y_1;

                weight_deform_gpu[x + y*kernel_size + i] = val;
            }
        }

        // compensate for dropped items
        const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
        for (int y = 0; y < kernel_size; ++y) {
            for (int x = 0; x < kernel_size; ++x) {
                weight_deform_gpu[x + y*kernel_size + i] *= coef;
            }
        }
    }
}


extern "C" void smooth_rotate_weights_gpu(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int size, int angle, int reverse)
{
    const int kernel_area = size*size;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(nweights / kernel_area, block_size);
    smooth_rotate_weights_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_weight_gpu, weight_deform_gpu, nweights, n, size, angle, reverse);

    CHECK_CUDA(cudaPeekAtLastError());
}



__global__  void stretch_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int kernel_size, float scale, int reverse)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int kernel_area = kernel_size * kernel_size;
    const int i = index * kernel_area;

    const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
    const int stage_id = index / stage_step;

    // nweights = (c / groups) * n * size * size;
    // kernel_area = size*size

    if (i < nweights)
    {

        if (stage_id == 0) {
            // simple copy
            for (int x = 0; x < kernel_size; ++x) {
                for (int y = 0; y < kernel_size; ++y) {
                    weight_deform_gpu[x + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                }
            }
        }
        else if (stage_id > 0)
        {
            if (stage_id == 1) scale = 0.65;
            else if (stage_id == 2) scale = 0.8;
            else if (stage_id == 3) scale = 1.3;

            if (reverse) scale = 1 / scale;

            const int x_c = kernel_size / 2;
            const int y_c = kernel_size / 2;

            float dropout_sum = 0;

            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    // Xsource = x_c + (x_d - x_c) / scale
                    // Ysource = y_c + (y_d - y_c) / scale

                    float x_s = x_c + (x - x_c) / scale;
                    float y_s = y_c + (y - y_c) / scale;

                    int x_0 = floor(x_s);   // round down
                    int x_1 = ceil(x_s);    // round up
                    if (x_0 == x_1) x_1 = x_0 + 1;
                    int y_0 = floor(y_s);
                    int y_1 = ceil(y_s);
                    if (y_0 == y_1) y_1 = y_0 + 1;

                    float c_x_0 = x_1 - x_s;
                    float c_x_1 = x_s - x_0;
                    float c_y_0 = y_1 - y_s;
                    float c_y_1 = y_s - y_0;

                    float val = 0;
                    if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                    else dropout_sum += c_x_0 * c_y_0;

                    if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                    else dropout_sum += c_x_1 * c_y_0;

                    if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                    else dropout_sum += c_x_0 * c_y_1;

                    if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                    else dropout_sum += c_x_1 * c_y_1;

                    weight_deform_gpu[x + y*kernel_size + i] = val;
                }
            }

            // compensate for dropped items
            //const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    //if (scale < 1) weight_deform_gpu[x + y*kernel_size + i] /= scale;// *= coef;
                    weight_deform_gpu[x + y*kernel_size + i] /= scale;// *= coef;
                }
            }
        }
    }
}


extern "C" void stretch_weights_gpu(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int size, float scale, int reverse)
{
    const int kernel_area = size*size;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(nweights / kernel_area, block_size);
    stretch_weights_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_weight_gpu, weight_deform_gpu, nweights, n, size, scale, reverse);

    CHECK_CUDA(cudaPeekAtLastError());
}



__global__  void sway_and_flip_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int kernel_size, int angle, int reverse)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int kernel_area = kernel_size * kernel_size;
    const int i = index * kernel_area;

    const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
    const int stage_id = index / stage_step;

    // nweights = (c / groups) * n * size * size;
    // kernel_area = size*size

    if (i < nweights)
    {

        if (stage_id == 0) {
            // simple copy
            for (int x = 0; x < kernel_size; ++x) {
                for (int y = 0; y < kernel_size; ++y) {
                    weight_deform_gpu[x + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                }
            }
        }
        else if (stage_id == 1 || stage_id == 2)
        {
            // rotate left or right
            if (stage_id == 2) angle = -angle;
            if (reverse) angle = -angle;

            const float cos_a = cosf(angle * 3.14159265 / 180);
            const float sin_a = sinf(angle * 3.14159265 / 180);
            const int x_c = kernel_size / 2;
            const int y_c = kernel_size / 2;

            float dropout_sum = 0;

            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    // Xsource = x*cos(alpha) + y*sin(alpha)
                    // Ysource = -x*sin(alpha) + y*cos(alpha)

                    float x_s = x_c + (x - x_c)*cos_a + (y - y_c)*sin_a;
                    float y_s = y_c - (x - x_c)*sin_a + (y - y_c)*cos_a;

                    int x_0 = floor(x_s);   // round down
                    int x_1 = ceil(x_s);    // round up
                    if (x_0 == x_1) x_1 = x_0 + 1;
                    int y_0 = floor(y_s);
                    int y_1 = ceil(y_s);
                    if (y_0 == y_1) y_1 = y_0 + 1;

                    float c_x_0 = x_1 - x_s;
                    float c_x_1 = x_s - x_0;
                    float c_y_0 = y_1 - y_s;
                    float c_y_1 = y_s - y_0;

                    float val = 0;
                    if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                    else dropout_sum += c_x_0 * c_y_0;

                    if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                    else dropout_sum += c_x_1 * c_y_0;

                    if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                    else dropout_sum += c_x_0 * c_y_1;

                    if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                    else dropout_sum += c_x_1 * c_y_1;

                    weight_deform_gpu[x + y*kernel_size + i] = val;
                }
            }

            // compensate for dropped items
            const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    weight_deform_gpu[x + y*kernel_size + i] *= coef;
                }
            }
        }
        else if (stage_id == 3)
        {
            // flip
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    weight_deform_gpu[(kernel_size - x - 1) + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                }
            }
        }
    }
}


extern "C" void sway_and_flip_weights_gpu(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int size, int angle, int reverse)
{
    const int kernel_area = size*size;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(nweights / kernel_area, block_size);
    sway_and_flip_weights_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_weight_gpu, weight_deform_gpu, nweights, n, size, angle, reverse);

    CHECK_CUDA(cudaPeekAtLastError());
}







__global__  void rotate_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int kernel_size, int reverse)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int kernel_area = kernel_size * kernel_size;
    const int i = index * kernel_area;

    const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
    const int stage_id = index / stage_step;

    // nweights = (c / groups) * n * size * size;
    // kernel_area = size*size

    if (i < nweights)
    {
        // if(reverse)

        if (stage_id == 0) {
            // simple copy
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    const int src_i = x + y*kernel_size + i;
                    const int dst_i = x + y*kernel_size + i;
                    if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                    else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
                }
            }
        }
        else if (stage_id == 1)
        {
            // 90 degree clockwise rotation - 1
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    const int src_i = x + y*kernel_size + i;
                    const int dst_i = (kernel_size - 1 - y) + x*kernel_size + i;
                    if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                    else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
                }
            }
        }
        else if (stage_id == 2)
        {
            // 180 degree clockwise rotation - 2
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    const int src_i = x + y*kernel_size + i;
                    const int dst_i = (kernel_size - 1 - x) + (kernel_size - 1 - y)*kernel_size + i;
                    if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                    else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
                }
            }
        }
        else if (stage_id == 3)
        {
            // 270 degree clockwise rotation - 3
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    const int src_i = x + y*kernel_size + i;
                    const int dst_i = y + (kernel_size - 1 - x)*kernel_size + i;
                    if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                    else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
                }
            }
        }
    }
}


extern "C" void rotate_weights_gpu(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int size, int reverse)
{
    const int kernel_area = size*size;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(nweights / kernel_area, block_size);
    rotate_weights_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_weight_gpu, weight_deform_gpu, nweights, n, size, reverse);

    CHECK_CUDA(cudaPeekAtLastError());
}



__global__  void stretch_sway_flip_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int kernel_size, float angle, int reverse)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int kernel_area = kernel_size * kernel_size;
    const int i = index * kernel_area;

    const int stage_step = (nweights / kernel_area) / 8;  // 8 stages
    const int stage_id = index / stage_step;

    // nweights = (c / groups) * n * size * size;
    // kernel_area = size*size

    if (i < nweights)
    {

        if (stage_id == 0) {
            // simple copy
            for (int x = 0; x < kernel_size; ++x) {
                for (int y = 0; y < kernel_size; ++y) {
                    weight_deform_gpu[x + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                }
            }
        }
        else if (stage_id == 1 || stage_id == 2 || stage_id == 3 || stage_id == 4)
        {
            float scale = 0.5;
            if (stage_id == 1) scale = 0.65;
            else if (stage_id == 2) scale = 0.8;
            else if (stage_id == 3) scale = 1.2;
            else if (stage_id == 4) scale = 1.4;

            if (reverse) scale = 1 / scale;

            const int x_c = kernel_size / 2;
            const int y_c = kernel_size / 2;

            float dropout_sum = 0;

            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    // Xsource = x_c + (x_d - x_c) / scale
                    // Ysource = y_c + (y_d - y_c) / scale

                    float x_s = x_c + (x - x_c) / scale;
                    float y_s = y_c + (y - y_c) / scale;

                    int x_0 = floor(x_s);   // round down
                    int x_1 = ceil(x_s);    // round up
                    if (x_0 == x_1) x_1 = x_0 + 1;
                    int y_0 = floor(y_s);
                    int y_1 = ceil(y_s);
                    if (y_0 == y_1) y_1 = y_0 + 1;

                    float c_x_0 = x_1 - x_s;
                    float c_x_1 = x_s - x_0;
                    float c_y_0 = y_1 - y_s;
                    float c_y_1 = y_s - y_0;

                    float val = 0;
                    if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                    else dropout_sum += c_x_0 * c_y_0;

                    if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                    else dropout_sum += c_x_1 * c_y_0;

                    if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                    else dropout_sum += c_x_0 * c_y_1;

                    if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                    else dropout_sum += c_x_1 * c_y_1;

                    weight_deform_gpu[x + y*kernel_size + i] = val;
                }
            }

            // compensate for dropped items
            //const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    if(scale > 1)
                        weight_deform_gpu[x + y*kernel_size + i] /= scale;// *= coef;
                }
            }
        }
        else if (stage_id == 5 || stage_id == 6)
        {
            // rotate left or right
            if (stage_id == 6) angle = -angle;
            if (reverse) angle = -angle;

            const float cos_a = cosf(angle * 3.14159265 / 180);
            const float sin_a = sinf(angle * 3.14159265 / 180);
            const int x_c = kernel_size / 2;
            const int y_c = kernel_size / 2;

            float dropout_sum = 0;

            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    // Xsource = x*cos(alpha) + y*sin(alpha)
                    // Ysource = -x*sin(alpha) + y*cos(alpha)

                    float x_s = x_c + (x - x_c)*cos_a + (y - y_c)*sin_a;
                    float y_s = y_c - (x - x_c)*sin_a + (y - y_c)*cos_a;

                    int x_0 = floor(x_s);   // round down
                    int x_1 = ceil(x_s);    // round up
                    if (x_0 == x_1) x_1 = x_0 + 1;
                    int y_0 = floor(y_s);
                    int y_1 = ceil(y_s);
                    if (y_0 == y_1) y_1 = y_0 + 1;

                    float c_x_0 = x_1 - x_s;
                    float c_x_1 = x_s - x_0;
                    float c_y_0 = y_1 - y_s;
                    float c_y_1 = y_s - y_0;

                    float val = 0;
                    if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                    else dropout_sum += c_x_0 * c_y_0;

                    if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                    else dropout_sum += c_x_1 * c_y_0;

                    if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                    else dropout_sum += c_x_0 * c_y_1;

                    if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                    else dropout_sum += c_x_1 * c_y_1;

                    weight_deform_gpu[x + y*kernel_size + i] = val;
                }
            }

            // compensate for dropped items
            const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    weight_deform_gpu[x + y*kernel_size + i] *= coef;
                }
            }
        }
        else if (stage_id == 7)
        {
            // flip
            for (int y = 0; y < kernel_size; ++y) {
                for (int x = 0; x < kernel_size; ++x) {
                    weight_deform_gpu[(kernel_size - x - 1) + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                }
            }
        }
    }
}


extern "C" void stretch_sway_flip_weights_gpu(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int size, int angle, int reverse)
{
    const int kernel_area = size*size;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(nweights / kernel_area, block_size);
    stretch_sway_flip_weights_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_weight_gpu, weight_deform_gpu, nweights, n, size, angle, reverse);

    CHECK_CUDA(cudaPeekAtLastError());
}



__global__  void reduce_and_expand_array_kernel(const float *src_gpu, float *dst_gpu, int current_size, int groups)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < current_size) {
        float val = 0;
        for (int i = 0; i < groups; ++i) {
            val += src_gpu[index + i*current_size];
        }
        for (int i = 0; i < groups; ++i) {
            dst_gpu[index + i*current_size] = val / groups;
        }
    }
}

extern "C" void reduce_and_expand_array_gpu(const float *src_gpu, float *dst_gpu, int size, int groups)
{
    const int current_size = size / groups;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(current_size, block_size);
    reduce_and_expand_array_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_gpu, dst_gpu, current_size, groups);

    CHECK_CUDA(cudaPeekAtLastError());
}



__global__  void expand_array_kernel(const float *src_gpu, float *dst_gpu, int current_size, int groups)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < current_size) {
        for (int i = 0; i < groups; ++i) {
            dst_gpu[index + i*current_size] = src_gpu[index];
        }
    }
}

extern "C" void expand_array_gpu(const float *src_gpu, float *dst_gpu, int size, int groups)
{
    const int current_size = size / groups;
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(current_size, block_size);
    expand_array_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_gpu, dst_gpu, current_size, groups);

    CHECK_CUDA(cudaPeekAtLastError());
}



__global__  void mult_inverse_array_kernel(const float *src_gpu, float *dst_gpu, int size, const float eps)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < size) {
        float val = src_gpu[index];
        float sign = (val < 0) ? -1 : 1;
        // eps = 1 by default
        // eps = 2 - lower delta
        // eps = 0 - higher delta (linear)
        // eps = -1 - high delta (inverse number)
        dst_gpu[index] = powf(fabs(val), eps) * sign;
    }
}

extern "C" void mult_inverse_array_gpu(const float *src_gpu, float *dst_gpu, int size, float eps)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    mult_inverse_array_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (src_gpu, dst_gpu, size, eps);

    CHECK_CUDA(cudaPeekAtLastError());
}



__global__ void P_constrastive_f_det_kernel(int *labels, unsigned int feature_size, float temperature, contrastive_params *contrast_p, const int contrast_p_size)
{
    const int il = blockIdx.x*blockDim.x + threadIdx.x;

    if (il < contrast_p_size) {
        const float sim = contrast_p[il].sim;
        const size_t i = contrast_p[il].i;
        const size_t j = contrast_p[il].j;

        const float numerator = expf(sim / temperature);

        float denominator = 0;
        int k;
        for (k = 0; k < contrast_p_size; ++k) {
            contrastive_params cp = contrast_p[k];
            //if (k != i && labels[k] != labels[i]) {
            //if (k != i) {
            if (cp.i != i && cp.j == j) {
                //const float sim_den = cp.sim;
                ////const float sim_den = find_sim(k, l, contrast_p, contrast_p_size); // cosine_similarity(z[k], z[l], feature_size);
                //denominator += expf(sim_den / temperature);
                denominator += cp.exp_sim;
            }
        }

        float result = 0.9999;
        if (denominator != 0) result = numerator / denominator;
        if (result > 1) result = 0.9999;

        contrast_p[il].P = result;
    }
}


extern "C" void P_constrastive_f_det_gpu(int *labels, unsigned int feature_size, float temperature, contrastive_params *contrast_p, const int contrast_p_size)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(contrast_p_size, block_size);
    P_constrastive_f_det_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (labels, feature_size, temperature, contrast_p, contrast_p_size);

    CHECK_CUDA(cudaPeekAtLastError());
}