#include "darknet.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <float.h>

#include "activations.h"
#include "dark_cuda.h"

__device__ float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
__device__ float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

__device__ float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
__device__ float linear_activate_kernel(float x){return x;}
__device__ float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
__device__ float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}
__device__ float relu_activate_kernel(float x){return x*(x>0);}
__device__ float relu6_activate_kernel(float x) { return min_val_cmp(max_val_cmp(x, 0), 6); }
__device__ float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
__device__ float selu_activate_kernel(float x) { return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x) - 1); }
__device__ float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
__device__ float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
__device__ float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}
__device__ float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}
__device__ float gelu_activate_kernel(float x){return (0.5*x*(1 + tanhf(0.797885*x + 0.035677*powf(x, 3))));}
__device__ float softplus_kernel(float x, float threshold = 20) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return log1pf(expf(x));
    //return logf(expf(x) + 1);
}
__device__ float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
__device__ float stair_activate_kernel(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floorf(x/2.f);
    else return (x - n) + floorf(x/2.f);
}


__device__ float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
__device__ float linear_gradient_kernel(float x){return 1;}
__device__ float logistic_gradient_kernel(float x){return (1-x)*x;}
__device__ float loggy_gradient_kernel(float x)
{
    float y = (x+1.F)/2.F;
    return 2*(1-y)*y;
}
__device__ float relu_gradient_kernel(float x){return (x>0);}
__device__ float relu6_gradient_kernel(float x) { return (x > 0 && x < 6); }
__device__ float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
__device__ float selu_gradient_kernel(float x) { return (x >= 0)*1.0507f + (x < 0)*(x + 1.0507f*1.6732f); }
__device__ float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
__device__ float ramp_gradient_kernel(float x){return (x>0)+.1f;}
__device__ float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
__device__ float tanh_gradient_kernel(float x){return 1-x*x;}
__device__ float sech_gpu(float x) { return 2 / (expf(x) + expf(-x)); }
__device__ float gelu_gradient_kernel(float x) {
    const float x3 = powf(x, 3);
    return 0.5*tanhf(0.0356774*x3 + 0.797885*x) + (0.0535161*x3 + 0.398942*x) * powf(sech_gpu(0.0356774*x3 + 0.797885*x), 2) + 0.5;
}
__device__ float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}
__device__ float stair_gradient_kernel(float x)
{
    if (floorf(x) == x) return 0;
    return 1;
}

__device__ float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case RELU6:
            return relu6_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case SELU:
            return selu_activate_kernel(x);
        case GELU:
            return gelu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

__device__ float gradient_kernel(float x, ACTIVATION a)
{
    switch (a) {
    case LINEAR:
        return linear_gradient_kernel(x);
    case LOGISTIC:
        return logistic_gradient_kernel(x);
    case LOGGY:
        return loggy_gradient_kernel(x);
    case RELU:
        return relu_gradient_kernel(x);
    case RELU6:
        return relu6_gradient_kernel(x);
    case NORM_CHAN:
        return relu_gradient_kernel(x);
    case ELU:
        return elu_gradient_kernel(x);
    case SELU:
        return selu_gradient_kernel(x);
    case GELU:
        return gelu_gradient_kernel(x);
    case RELIE:
        return relie_gradient_kernel(x);
    case RAMP:
        return ramp_gradient_kernel(x);
    case LEAKY:
        return leaky_gradient_kernel(x);
    case TANH:
        return tanh_gradient_kernel(x);
    case PLSE:
        return plse_gradient_kernel(x);
    case STAIR:
        return stair_gradient_kernel(x);
    case HARDTAN:
        return hardtan_gradient_kernel(x);
    case LHTAN:
        return lhtan_gradient_kernel(x);
    }
    return 0;
}

__global__ void binary_gradient_array_kernel(float *x, float *dy, int n, int s, BINARY_ACTIVATION a, float *dx)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s / 2 + i];
    if (id < n) {
        float de = dy[id];
        dx[b*s + i] = x2*de;
        dx[b*s + s / 2 + i] = x1*de;
    }
}

extern "C" void binary_gradient_array_gpu(float *x, float *dx, int n, int size, BINARY_ACTIVATION a, float *y)
{
    binary_gradient_array_kernel << <cuda_gridsize(n / 2), BLOCK, 0, get_cuda_stream() >> >(x, dx, n / 2, size, a, y);
    CHECK_CUDA(cudaPeekAtLastError());
}
__global__ void binary_activate_array_kernel(float *x, int n, int s, BINARY_ACTIVATION a, float *y)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s / 2 + i];
    if (id < n) y[id] = x1*x2;
}

extern "C" void binary_activate_array_gpu(float *x, int n, int size, BINARY_ACTIVATION a, float *y)
{
    binary_activate_array_kernel << <cuda_gridsize(n / 2), BLOCK, 0, get_cuda_stream() >> >(x, n / 2, size, a, y);
    CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void activate_array_kernel(float *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_kernel(x[i], a);
}



__global__ void activate_array_swish_kernel(float *x, int n, float *output_sigmoid_gpu, float *output_gpu)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x_val = x[i];
        float sigmoid = logistic_activate_kernel(x_val);
        if (output_sigmoid_gpu) output_sigmoid_gpu[i] = sigmoid;
        output_gpu[i] = x_val * sigmoid;
    }
}

__device__ float mish_njuffa(float x)
{
    float r;
    float e = expf(x);
    r = 1.0f / fmaf(fmaf(-0.5f, e, -1.0f), e, -1.0f);
    r = fmaf(r, x, x);
    return r;
}

__device__ float mish_yashas(float x)
{
    float e = __expf(x);
    if (x <= -18.0f)
        return x * e;

    float n = e * e + 2 * e;
    if (x <= -5.0f)
        return x * __fdividef(n, n + 2);

    return x - 2 * __fdividef(x, n + 2);
}

__device__ float mish_yashas2(float x)
{
    float e = __expf(x);
    float n = e * e + 2 * e;
    if (x <= -0.6f)
        return x * __fdividef(n, n + 2);

    return x - 2 * __fdividef(x, n + 2);
}

// https://github.com/digantamisra98/Mish
__global__ void activate_array_mish_kernel(float *x, int n, float *activation_input, float *output_gpu)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float MISH_THRESHOLD = 20;
        float x_val = x[i];
        if (activation_input) activation_input[i] = x_val;    // store value before activation
        //output_gpu[i] = x_val * tanh_activate_kernel(logf(1 + expf(x_val)));

        // Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L17-L20
        // TF: https://github.com/tensorflow/addons/blob/093cdfa85d334cbe19a37624c33198f3140109ed/tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.h#L40-L49
        // log1p(x) == log(x + 1)
        //output_gpu[i] = x_val * tanh_activate_kernel( softplus_kernel(x_val, MISH_THRESHOLD) );
        output_gpu[i] = mish_yashas2(x_val);
        //output_gpu[i] = mish_njuffa(x_val);
    }
}

__device__ float hard_mish_yashas(float x)
{
    if (x > 0)
        return x;
    if (x > -2)
        return x * x / 2 + x;
    return 0;
}

__global__ void activate_array_hard_mish_kernel(float *x, int n, float *activation_input, float *output_gpu)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {

        float x_val = x[i];
        if (activation_input) activation_input[i] = x_val;    // store value before activation
        output_gpu[i] = hard_mish_yashas(x_val);
    }
}
__global__ void activate_array_leaky_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = leaky_activate_kernel(x[index]);
    }
}

__global__ void activate_array_selu_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = selu_activate_kernel(x[index]);
    }
}

__global__ void activate_array_gelu_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = gelu_activate_kernel(x[index]);
    }
}

__global__ void activate_array_logistic_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = logistic_activate_kernel(x[index]);
    }
}

__global__ void activate_array_tanh_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = tanh_activate_kernel(x[index]);
    }
}

__global__ void activate_array_hardtan_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = hardtan_activate_kernel(x[index]);
    }
}

__global__ void activate_array_relu_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = relu_activate_kernel(x[index]);
    }
}

__global__ void activate_array_relu6_kernel(float *x, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = relu6_activate_kernel(x[index]);
    }
}

__global__ void gradient_array_kernel(float *x, int n, ACTIVATION a, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) delta[i] *= gradient_kernel(x[i], a);
}

// https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cu#L28-L30
__global__ void gradient_array_swish_kernel(float *x, int n, float *sigmoid_gpu, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float swish = x[i];
        delta[i] *= swish + sigmoid_gpu[i] * (1 - swish); // gradient_kernel(x[i], a);
    }
}

// https://github.com/digantamisra98/Mish
__global__ void gradient_array_mish_kernel(int n, float *activation_input_gpu, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float MISH_THRESHOLD = 20.0f;

        // implementation from TensorFlow: https://github.com/tensorflow/addons/blob/093cdfa85d334cbe19a37624c33198f3140109ed/tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.h#L66-L80
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        // log1p(x) == log(x + 1)
        const float inp = activation_input_gpu[i];
        const float sp = softplus_kernel(inp, MISH_THRESHOLD);
        const float grad_sp = -expm1f(-sp);
        //const float grad_sp = 1 - expf(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        delta[i] *= grad;

        //float x = activation_input[i];
        //float d = 2 * expf(x) + expf(2 * x) + 2;
        //float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x)*(4 * x + 6);
        //float derivative = expf(x) * w / (d * d);
        //delta[i] *= derivative;
    }
}

__device__ float hard_mish_yashas_grad(float x)
{
    if (x > 0)
        return 1;
    if (x > -2)
        return x + 1;
    return 0;
}

__global__ void gradient_array_hard_mish_kernel(int n, float *activation_input_gpu, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {

        const float x = activation_input_gpu[i];
        delta[i] *= hard_mish_yashas_grad(x);
    }
}

__global__ void gradient_array_leaky_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= leaky_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_revleaky_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] /= leaky_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_selu_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= selu_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_gelu_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= gelu_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_logistic_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= logistic_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_tanh_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= tanh_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_hardtan_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= hardtan_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_relu_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= relu_gradient_kernel(x[index]);
    }
}

__global__ void gradient_array_relu6_kernel(float *x, int n, float *delta)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < n) {
        delta[index] *= relu6_gradient_kernel(x[index]);
    }
}

extern "C" void activate_array_ongpu(float *x, int n, ACTIVATION a)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    if (a == LINEAR) return;
    else if (a == LEAKY || a == REVLEAKY) activate_array_leaky_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else if (a == LOGISTIC) activate_array_logistic_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else if (a == TANH) activate_array_tanh_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else if (a == HARDTAN) activate_array_hardtan_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else if (a == RELU) activate_array_relu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else if (a == RELU6) activate_array_relu6_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else if (a == SELU) activate_array_selu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else if (a == GELU) activate_array_gelu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n);
    else
        activate_array_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream()>>>(x, n, a);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void activate_array_swish_ongpu(float *x, int n, float *output_sigmoid_gpu, float *output_gpu)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    activate_array_swish_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(x, n, output_sigmoid_gpu, output_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void activate_array_mish_ongpu(float *x, int n, float *activation_input_gpu, float *output_gpu)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    activate_array_mish_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(x, n, activation_input_gpu, output_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void activate_array_hard_mish_ongpu(float *x, int n, float *activation_input_gpu, float *output_gpu)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    activate_array_hard_mish_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(x, n, activation_input_gpu, output_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void gradient_array_ongpu(float *x, int n, ACTIVATION a, float *delta)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    if (a == LINEAR) return;
    else if (a == LEAKY) gradient_array_leaky_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == REVLEAKY) gradient_array_revleaky_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == LOGISTIC) gradient_array_logistic_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == TANH) gradient_array_tanh_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == HARDTAN) gradient_array_hardtan_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == RELU) gradient_array_relu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == RELU6) gradient_array_relu6_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    //else if (a == NORM_CHAN) gradient_array_relu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == NORM_CHAN_SOFTMAX || a == NORM_CHAN) {
        printf(" Error: should be used custom NORM_CHAN_SOFTMAX-function for gradient \n");
        exit(0);
    }
    else if (a == SELU) gradient_array_selu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else if (a == GELU) gradient_array_gelu_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> >(x, n, delta);
    else
        gradient_array_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (x, n, a, delta);
    CHECK_CUDA(cudaPeekAtLastError());
}


extern "C" void gradient_array_swish_ongpu(float *x, int n, float *sigmoid_gpu, float *delta)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    gradient_array_swish_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (x, n, sigmoid_gpu, delta);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void gradient_array_mish_ongpu(int n, float *activation_input_gpu, float *delta)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    gradient_array_mish_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (n, activation_input_gpu, delta);
    CHECK_CUDA(cudaPeekAtLastError());
}

extern "C" void gradient_array_hard_mish_ongpu(int n, float *activation_input_gpu, float *delta)
{
    const int num_blocks = get_number_of_blocks(n, BLOCK);
    gradient_array_hard_mish_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (n, activation_input_gpu, delta);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void activate_array_normalize_channels_kernel(float *x, int size, int batch, int channels, int wh_step, float *output_gpu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int wh_i = i % wh_step;
    int b = i / wh_step;

    const float eps = 0.0001;
    if (i < size) {
        float sum = eps;
        int k;
        for (k = 0; k < channels; ++k) {
            float val = x[wh_i + k * wh_step + b*wh_step*channels];
            if (val > 0) sum += val;
        }
        for (k = 0; k < channels; ++k) {
            float val = x[wh_i + k * wh_step + b*wh_step*channels];
            if (val > 0) val = val / sum;
            else val = 0;
            output_gpu[wh_i + k * wh_step + b*wh_step*channels] = val;
        }
    }
}

extern "C" void activate_array_normalize_channels_ongpu(float *x, int n, int batch, int channels, int wh_step, float *output_gpu)
{
    // n = w*h*c*batch
    // size = w*h*batch
    int size = n / channels;

    const int num_blocks = get_number_of_blocks(size, BLOCK);

    activate_array_normalize_channels_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (x, size, batch, channels, wh_step, output_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}



__global__ void activate_array_normalize_channels_softmax_kernel(float *x, int size, int batch, int channels, int wh_step, float *output_gpu, int use_max_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int wh_i = i % wh_step;
    int b = i / wh_step;

    const float eps = 0.0001;
    if (i < size) {
        float sum = eps;
        float max_val = -FLT_MAX;
        int k;
        if (use_max_val) {
            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                if (val > max_val || k == 0) max_val = val;
            }
        }
        else
            max_val = 0;

        for (k = 0; k < channels; ++k) {
            float val = x[wh_i + k * wh_step + b*wh_step*channels];
            sum += expf(val - max_val);
        }
        for (k = 0; k < channels; ++k) {
            float val = x[wh_i + k * wh_step + b*wh_step*channels];
            val = expf(val - max_val) / sum;
            if (isnan(val) || isinf(val)) val = 0;
            output_gpu[wh_i + k * wh_step + b*wh_step*channels] = val;
        }
    }
}

extern "C" void activate_array_normalize_channels_softmax_ongpu(float *x, int n, int batch, int channels, int wh_step, float *output_gpu, int use_max_val)
{
    // n = w*h*c*batch
    // size = w*h*batch
    int size = n / channels;

    const int num_blocks = get_number_of_blocks(size, BLOCK);

    activate_array_normalize_channels_softmax_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (x, size, batch, channels, wh_step, output_gpu, use_max_val);
    CHECK_CUDA(cudaPeekAtLastError());
}



__global__ void gradient_array_normalize_channels_softmax_kernel(float *x, int size, int batch, int channels, int wh_step, float *delta_gpu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int wh_i = i % wh_step;
    int b = i / wh_step;

    if (i < size) {
        int k;
        /*
        float grad = 0;
        for (k = 0; k < channels; ++k) {
            const int index = wh_i + k * wh_step + b*wh_step*channels;
            float out = x[index];
            float delta = delta_gpu[index];
            grad += out*fabs(delta);
        }
        */
        for (k = 0; k < channels; ++k) {
            const int index = wh_i + k * wh_step + b*wh_step*channels;
            float delta = delta_gpu[index];
            float grad = x[index] * (1 - x[index]);
            delta = delta * grad;
            if (isnan(delta) || isinf(delta)) delta = 0;
            delta_gpu[index] = delta;
        }
    }
}

extern "C" void gradient_array_normalize_channels_softmax_ongpu(float *output_gpu, int n, int batch, int channels, int wh_step, float *delta_gpu)
{
    // n = w*h*c*batch
    // size = w*h*batch
    int size = n / channels;

    const int num_blocks = get_number_of_blocks(size, BLOCK);

    gradient_array_normalize_channels_softmax_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (output_gpu, size, batch, channels, wh_step, delta_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void gradient_array_normalize_channels_kernel(float *x, int size, int batch, int channels, int wh_step, float *delta_gpu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int wh_i = i % wh_step;
    int b = i / wh_step;

    if (i < size) {
        int k;
        /*
        float grad = 0;
        for (k = 0; k < channels; ++k) {
            const int index = wh_i + k * wh_step + b*wh_step*channels;
            float out = x[index];
            float delta = delta_gpu[index];
            grad += out*fabs(delta);
        }
        */
        for (k = 0; k < channels; ++k) {
            const int index = wh_i + k * wh_step + b*wh_step*channels;
            if (x[index] > 0) {
                float delta = delta_gpu[index];
                float grad = x[index];
                delta = delta * grad;
                delta_gpu[index] = delta;
            }
        }
    }
}

extern "C" void gradient_array_normalize_channels_ongpu(float *output_gpu, int n, int batch, int channels, int wh_step, float *delta_gpu)
{
    // n = w*h*c*batch
    // size = w*h*batch
    int size = n / channels;

    const int num_blocks = get_number_of_blocks(size, BLOCK);

    gradient_array_normalize_channels_kernel << <num_blocks, BLOCK, 0, get_cuda_stream() >> > (output_gpu, size, batch, channels, wh_step, delta_gpu);
    CHECK_CUDA(cudaPeekAtLastError());
}