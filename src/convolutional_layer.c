#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")
#endif

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifndef AI2
#define AI2 0
void forward_xnor_layer(layer l, network_state state);
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

    #ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
    #endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean: -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}

size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
    #endif
    if(l.xnor) return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l, int cudnn_preference)
{

// CUDNN_HALF
    // TRUE_HALF_CONFIG is only supported on architectures with true fp16 support (compute capability 5.3 and 6.0):
    //   Tegra X1, Jetson TX1, DRIVE CX, DRIVE PX, Quadro GP100, Tesla P100
    // PSEUDO_HALF_CONFIG is required for Tensor Cores - our case!

    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

#if(CUDNN_MAJOR >= 7)
    // Tensor Core uses CUDNN_TENSOR_OP_MATH instead of CUDNN_DEFAULT_MATH
    // For *_ALGO_WINOGRAD_NONFUSED can be used CUDNN_DATA_FLOAT
    // otherwise Input, Filter and Output descriptors (xDesc, yDesc, wDesc, dxDesc, dyDesc and dwDesc as applicable) have dataType = CUDNN_DATA_HALF
    // Three techniques for training using Mixed-precision: https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
    // 1. Accumulation into FP32
    // 2. Loss Scaling - required only for: activation gradients. We do not use.
    // 3. FP32 Master Copy of Weights
    // More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops
    cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH);
#if((CUDNN_MAJOR*10 + CUDNN_MINOR) >= 72)   // cuDNN >= 7.2
    cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
#endif
#endif

    // INT8_CONFIG, INT8_EXT_CONFIG, INT8x4_CONFIG and INT8x4_EXT_CONFIG are only supported
    //   on architectures with DP4A support (compute capability 6.1 and later).
    //cudnnDataType_t data_type = CUDNN_DATA_INT8;

    // backward delta
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w);
    cudnnSetFilter4dDescriptor(l->dweightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size);

    // forward
    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w);
    cudnnSetFilter4dDescriptor(l->weightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size);

#ifdef CUDNN_HALF
    // backward delta
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w);
    cudnnSetFilter4dDescriptor(l->dweightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size);

    // forward
    cudnnSetTensor4dDescriptor(l->srcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->dstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w);
    cudnnSetFilter4dDescriptor(l->weightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size);

    // batch norm
    cudnnSetTensor4dDescriptor(l->normDstTensorDescF16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w);
#endif

    // batch norm
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);
    cudnnSetTensor4dDescriptor(l->normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);

#if(CUDNN_MAJOR >= 6)
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);    // cudnn >= 6.0
#else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);    // cudnn 5.1
#endif
    int forward_algo = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    int backward_algo = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    int backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
    if (cudnn_preference == cudnn_smallest)
    {
        forward_algo = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
        backward_algo = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
        backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
        printf(" CUDNN-slow ");
    }

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            forward_algo,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            backward_algo,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            backward_filter,
            0,
            &l->bf_algo);

    //if (data_type == CUDNN_DATA_HALF)
    {
        // HALF-16 if(data_type == CUDNN_DATA_HALF)
        l->fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        l->bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        l->bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        // FLOAT-32 if(data_type == CUDNN_DATA_FLOAT)
        //l->fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        //l->bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
        //l->bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;

        int fw = 0, bd = 0, bf = 0;
        if (l->fw_algo16 == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) fw = 1;
            //printf("Tensor Cores - Forward enabled: l->fw_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM \n");
        if (l->fw_algo16 == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) fw = 2;
            //printf("Tensor Cores - Forward enabled: l->fw_algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED \n");

        if (l->bd_algo16 == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1) bd = 1;
            //printf("Tensor Cores - Backward-data enabled: l->bd_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1  \n");
        if (l->bd_algo16 == CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED) bd = 2;
            //printf("Tensor Cores - Backward-data enabled: l->bd_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED \n");

        if (l->bf_algo16 == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) bf = 1;
            //printf("Tensor Cores - Backward-filter enabled: l->bf_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1   \n");
        if (l->bf_algo16 == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED) bf = 2;
            //printf("Tensor Cores - Backward-filter enabled: l->bf_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED \n");

        //if (fw == 2 && bd == 2 && bf == 2) printf("TF ");
        //else if (fw == 1 && bd == 1 && bf == 1) printf("TH ");
    }
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.index = index;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = l.out_h*l.out_w;
        l.bit_align = src_align + (align - src_align % align);

        l.mean_arr = calloc(l.n, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.adam = 1;
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
        }

        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
#ifdef CUDNN_HALF
        l.weights_gpu16 = cuda_make_array(NULL, c*n*size*size / 2); //cuda_make_array(l.weights, c*n*size*size / 2);
        l.weight_updates_gpu16 = cuda_make_array(NULL, c*n*size*size / 2); //cuda_make_array(l.weight_updates, c*n*size*size / 2);
#endif
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
            l.mean_arr_gpu = cuda_make_array(0, l.n);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);

        cudnnCreateTensorDescriptor(&l.normDstTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);

        cudnnCreateTensorDescriptor(&l.normDstTensorDescF16);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc16);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc16);
        cudnnCreateFilterDescriptor(&l.weightDesc16);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc16);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc16);
        cudnnCreateFilterDescriptor(&l.dweightDesc16);

        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l, cudnn_fastest);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.bflops = (2.0 * l.n * l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else fprintf(stderr, "conv  ");
    fprintf(stderr, "%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    network_state state = {0};
    state.input = data;
    forward_convolutional_layer(l, state);
}

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    int old_w = l->w;
    int old_h = l->h;
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

    if (l->xnor) {
        //l->binary_input = realloc(l->inputs*l->batch, sizeof(float));
    }

#ifdef GPU
    if (old_w < w || old_h < h) {
        cuda_free(l->delta_gpu);
        cuda_free(l->output_gpu);

        l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
        l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

        if (l->batch_normalize) {
            cuda_free(l->x_gpu);
            cuda_free(l->x_norm_gpu);

            l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
            l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        }

        if (l->xnor) {
            cuda_free(l->binary_input_gpu);
            l->binary_input_gpu = cuda_make_array(0, l->inputs*l->batch);
        }
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l, cudnn_fastest);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);

#ifdef CUDNN
    // check for excessive memory consumption
    size_t free_byte;
    size_t total_byte;
    check_error(cudaMemGetInfo(&free_byte, &total_byte));
    if (l->workspace_size > free_byte || l->workspace_size >= total_byte / 2) {
        printf(" used slow CUDNN algo without Workspace! Need memory: %zu, available: %zu\n", l->workspace_size, (free_byte < total_byte/2) ? free_byte : total_byte/2);
        cudnn_convolutional_setup(l, cudnn_smallest);
        l->workspace_size = get_workspace_size(*l);
    }
#endif
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void gemm_nn_custom(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register float A_PART = ALPHA*A[i*lda + k];
            //printf("\n weight = %f \n", A_PART);
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}


void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
    size_t i, counter;
    counter = 0;
    for (i = 0; i < size; i += size / filters) {
        mean_arr[counter++] = fabs(src[i]);
    }
}

/*
void float_to_bit(float *src, unsigned char *dst, size_t size) {

    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);
    size_t i, dst_i, dst_shift;
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) set_bit(dst, i);
    }
}
*/

void bit_to_float(unsigned char *src, float *dst, size_t size, size_t filters, float *mean_arr) {
    memset(dst, 0, size *sizeof(float));
    size_t i,  src_i, src_shift;

    for (i = 0; i < size; ++i) {
        float mean_val = 1;
        if(mean_arr != NULL) mean_val = fabs(mean_arr[i / (size / filters)]);
        if(get_bit(src, i)) dst[i] = mean_val;
        else dst[i] = -mean_val;
    }
}

void binary_align_weights(convolutional_layer *l)
{
    int m = l->n;
    int k = l->size*l->size*l->c;
    size_t new_lda = k + (l->lda_align - k % l->lda_align); // (k / 8 + 1) * 8;
    l->new_lda = new_lda;

    binarize_weights(l->weights, m, k, l->binary_weights);

    size_t align_weights_size = new_lda * m;
    l->align_bit_weights_size = align_weights_size / 8 + 1;
    float *align_weights = calloc(align_weights_size, sizeof(float));
    l->align_bit_weights = calloc(l->align_bit_weights_size, sizeof(char));

    size_t i, j;
    // align A without transpose
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = l->binary_weights[i*k + j];
        }
    }
    float_to_bit(align_weights, l->align_bit_weights, align_weights_size);

    //l->mean_arr = calloc(l->n, sizeof(float));
    get_mean_array(align_weights, align_weights_size, l->n, l->mean_arr);

#ifdef GPU
    cudaError_t status;
    l->align_workspace_size = l->bit_align * l->size * l->size * l->c;
    status = cudaMalloc((void **)&l->align_workspace_gpu, l->align_workspace_size * sizeof(float));
    status = cudaMalloc((void **)&l->transposed_align_workspace_gpu, l->align_workspace_size * sizeof(float));
    check_error(status);

    //l->align_bit_weights_gpu = cuda_make_array(l->align_bit_weights, l->align_bit_weights_size * sizeof(char)/sizeof(float));
    status = cudaMalloc((void **)&l->align_bit_weights_gpu, l->align_bit_weights_size);
    check_error(status);
    status = cudaMemcpy(l->align_bit_weights_gpu, l->align_bit_weights, l->align_bit_weights_size, cudaMemcpyHostToDevice);
    check_error(status);
    status = cudaMemcpy(l->binary_weights_gpu, l->binary_weights, m*k*sizeof(float), cudaMemcpyHostToDevice);
    check_error(status);

    //l->mean_arr_gpu = cuda_make_array(l->mean_arr, l->n);
    cuda_push_array(l->mean_arr_gpu, l->mean_arr, l->n);
    cudaDeviceSynchronize();
#endif // GPU

    free(align_weights);
}

// binary transpose
size_t binary_transpose_align_input(int k, int n, float *b, char **t_bit_input, size_t ldb_align, int bit_align)
{
    size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
    //printf("\n n = %d, bit_align = %d \n", n, bit_align);
    size_t t_intput_size = new_ldb * bit_align;// n;
    size_t t_bit_input_size = t_intput_size / 8;// +1;

    *t_bit_input = calloc(t_bit_input_size, sizeof(char));
    int src_size = k * bit_align;

    // b - [bit_align, k] - [l.bit_align, l.size*l.size*l.c] = src_size
    // t_input - [bit_align, k] - [n', k]
    // t_bit_input - [new_ldb, n] - [k', n]

    //transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    transpose_bin(b, *t_bit_input, k, n, bit_align, new_ldb, 8);

    return t_intput_size;
}


void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        if (!l.align_bit_weights || state.train) {
            binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
            //printf("\n binarize_weights l.align_bit_weights = %p \n", l.align_bit_weights);
        }
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;

    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

    static int u = 0;
    u++;

    for(i = 0; i < l.batch; ++i){
        //im2col_cpu(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);

        //float *t_input = NULL;
        //if (l.xnor) {
        //    size_t new_ldb = k + (l.lda_align - k%l.lda_align);
        //    size_t t_intput_size = new_ldb * n;
        //    t_input = calloc(t_intput_size, sizeof(float));
        //    im2col_cpu_custom_transpose(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, t_input, new_ldb);
        //}
        //if (l.xnor && l.size == 3 && l.stride == 1 && l.pad == 1) {}
        //else
        // further optimizations: im2col_bin() for XNOR, and then transpose_aling_bin()
        //im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);


        //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        //gemm_nn_custom(m, n, k, 1, a, k, b, n, c, n);
        if (l.xnor && l.align_bit_weights && !state.train && (l.stride == 1 && l.pad == 1)) {
            memset(b, 0, l.bit_align*l.size*l.size*l.c * sizeof(float));
            //im2col_cpu_custom_align(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, l.bit_align);
            im2col_cpu_custom_bin(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, l.bit_align);

            size_t output_size = l.outputs;
            //float *count_output = calloc(output_size, sizeof(float));
            //size_t bit_output_size = output_size / 8 + 1;
            //char *bit_output = calloc(bit_output_size, sizeof(char));

            size_t intput_size = n * k; // (out_h*out_w) X (l.size*l.size*l.c) : after im2col()
            size_t bit_input_size = intput_size / 8 + 1;
            //char *bit_input = calloc(bit_input_size, sizeof(char));

            size_t weights_size = k * m; //l.size*l.size*l.c*l.n;
            size_t bit_weights_size = weights_size / 8 + 1;
            //char *bit_weights = calloc(bit_weights_size, sizeof(char));
            //float *mean_arr = calloc(l.n, sizeof(float));

            // test: float->bit->float
            //get_mean_array(l.weights, weights_size, l.n, mean_arr);
            //float_to_bit(l.weights, bit_weights, weights_size);
            //memset(l.weights, 0, weights_size * sizeof(float));
            //bit_to_float(bit_weights, l.weights, weights_size, l.n, mean_arr); // just for test float->bit->float

            //float_to_bit(b, bit_input, intput_size);
            //memset(b, 0, intput_size * sizeof(float));
            //bit_to_float(bit_input, b, intput_size, 1, NULL); // just for test float->bit->float

            // transpose B from NxK to KxN (x-axis (ldb = l.size*l.size*l.c) - should be multiple of 8 bits)
            {
                /*
                size_t ldb_align = 256;// 8;

                size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
                size_t t_intput_size = new_ldb * n;
                size_t t_bit_input_size = t_intput_size / 8;// +1;
                float *t_input = calloc(t_intput_size, sizeof(float));
                char *t_bit_input = calloc(t_bit_input_size, sizeof(char));

                //printf("\n bit_input_size = %d, n = %d, k = %d, ldb = %d \n", bit_input_size, n, k, n);
                //printf("\n t_bit_input_size = %d, k = %d, n = %d, new_ldb = %d \n", t_bit_input_size, k, n, new_ldb);


                //printf("\n align_weights_size = %d, k = %d, m = %d, lda = %d \n", align_weights_size, k, m, k);
                //printf("\n align_bit_weights_size = %d, k = %d, m = %d, new_lda = %d \n", align_bit_weights_size, k, m, new_ldb);


                // transpose and align B
                int i, j;
                for (i = 0; i < n; ++i) {
                    for (j = 0; j < k; ++j) {
                        t_input[i*new_ldb + j] = b[j*n + i];
                    }
                }
                float_to_bit(t_input, t_bit_input, t_intput_size);



                if (!l.align_bit_weights)
                {
                    size_t align_weights_size = new_ldb * m;
                    size_t align_bit_weights_size = align_weights_size / 8;// +1;
                    float *align_weights = calloc(align_weights_size, sizeof(float));
                    l.align_bit_weights = calloc(align_bit_weights_size, sizeof(char));

                    // align A without transpose
                    for (i = 0; i < m; ++i) {
                        for (j = 0; j < k; ++j) {
                            align_weights[i*new_ldb + j] = a[i*k + j];
                        }
                    }
                    float_to_bit(align_weights, l.align_bit_weights, align_weights_size);

                    l.mean_arr = calloc(l.n, sizeof(float));
                    get_mean_array(align_weights, align_weights_size, l.n, l.mean_arr);

                    free(align_weights);
                }
                */

                /*
                if (l.size == 3 && l.stride == 1 && l.pad == 1)
                {
                    //binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
                    //printf("\n mean = %f \n", l.mean_arr[0]);

                    convolution_2d(l.w, l.h, l.size, l.n, l.c, l.pad, l.stride,
                        //l.weights, state.input, l.output, l.mean_arr);
                        l.binary_weights, state.input, l.output, l.mean_arr);
                }
                else {
                    */

                    //size_t ldb_align = 256; // 256 bit for AVX2
                    int ldb_align = l.lda_align;
                    size_t new_ldb = k + (ldb_align - k%ldb_align);
                    char *t_bit_input = NULL;
                    size_t t_intput_size = binary_transpose_align_input(k, n, b, &t_bit_input, ldb_align, l.bit_align);
                    //char *t_bit_input = calloc(new_ldb * n, sizeof(char));    // for im2col_cpu_custom_transpose() only
                    //float_to_bit(t_input, t_bit_input, new_ldb * n);    // for im2col_cpu_custom_transpose() only

                    // 5x times faster than gemm()-float32
                    gemm_nn_custom_bin_mean_transposed(m, n, k, 1, l.align_bit_weights, new_ldb, t_bit_input, new_ldb, c, n, l.mean_arr);

                    //gemm_nn_custom_bin_mean_transposed(m, n, k, 1, bit_weights, k, t_bit_input, new_ldb, c, n, mean_arr);

                    //free(t_input);
                    free(t_bit_input);
                //}

            }

            // for bit_input: (k * n)
            //if (u == 8) gemm_nn_custom_bin_mean(m, n, k, 1, bit_weights, k, bit_input, n, c, n, mean_arr);  // last xnor layer
            //else gemm_nn_custom_bin_mean(m, n, k, 1, bit_weights, k, bit_input, n, c, n, NULL);

            //gemm_nn_custom_bin_mean(m, n, k, 1, bit_weights, k, bit_input, n, c, n, mean_arr);

            //printf("\n u = %d \n", u);

            //gemm_nn_custom(m, n, k, 1, a, k, b, n, c, n);

            //int j;
            //if (u != 8) for (j = 0; j < l.n; ++j) l.biases[j] = l.biases[j] / (mean_arr[j]*2);

            //free(count_output);
            //free(bit_input);
            //free(bit_weights);
            //free(mean_arr);
        }
        else {
            im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);

            gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            // bit-count to float
        }
        c += n*m;
        state.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

    //activate_array(l.output, m*n*l.batch, l.activation);
    activate_array_cpu_custom(l.output, m*n*l.batch, l.activation);

    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, state);
    }

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = state.workspace;
        float *c = l.weight_updates;

        float *im = state.input+i*l.c*l.h*l.w;

        im2col_cpu(im, l.c, l.h, l.w,
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(state.delta){
            a = l.weights;
            b = l.delta + i*m*k;
            c = state.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta+i*l.c*l.h*l.w);
        }
    }
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        //normalize_image(weights[i]);
    }
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

