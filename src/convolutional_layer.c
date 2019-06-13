#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
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

size_t get_workspace_size32(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s));
        if (s > most) most = s;
        return most;
    }
    #endif
    if (l.xnor) {
        size_t re_packed_input_size = l.c * l.w * l.h * sizeof(float);
        size_t workspace_size = (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
        if (workspace_size < re_packed_input_size) workspace_size = re_packed_input_size;
        return workspace_size;
    }
    return (size_t)l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);
}

size_t get_workspace_size16(layer l) {
#if defined(CUDNN) && defined(CUDNN_HALF)
    if (gpu_index >= 0) {
        size_t most = 0;
        size_t s = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc16,
            l.weightDesc16,
            l.convDesc,
            l.dstTensorDesc16,
            l.fw_algo16,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc16,
            l.ddstTensorDesc16,
            l.convDesc,
            l.dweightDesc16,
            l.bf_algo16,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
            l.weightDesc16,
            l.ddstTensorDesc16,
            l.convDesc,
            l.dsrcTensorDesc16,
            l.bd_algo16,
            &s));
        if (s > most) most = s;
        return most;
    }
#endif
    return 0;
    //if (l.xnor) return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
    //return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

size_t get_convolutional_workspace_size(layer l) {
    size_t workspace_size = get_workspace_size32(l);
    size_t workspace_size16 = get_workspace_size16(l);
    if (workspace_size16 > workspace_size) workspace_size = workspace_size16;
    return workspace_size;
}
#ifdef GPU
#ifdef CUDNN
void create_convolutional_cudnn_tensors(layer *l)
{
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normTensorDesc));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->srcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dstTensorDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->weightDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dsrcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->ddstTensorDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->dweightDesc));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDescF16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->srcTensorDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dstTensorDesc16));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->weightDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dsrcTensorDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->ddstTensorDesc16));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->dweightDesc16));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&l->convDesc));
}

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
    CHECK_CUDNN(cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH));
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(l->convDesc, l->groups));
#if((CUDNN_MAJOR*10 + CUDNN_MINOR) >= 72)   // cuDNN >= 7.2
    CHECK_CUDNN(cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif
#else   //if(CUDNN_MAJOR >= 7)
    if (l->groups > 1) {
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
#endif

    // INT8_CONFIG, INT8_EXT_CONFIG, INT8x4_CONFIG and INT8x4_EXT_CONFIG are only supported
    //   on architectures with DP4A support (compute capability 6.1 and later).
    //cudnnDataType_t data_type = CUDNN_DATA_INT8;

    // backward delta
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->dweightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

    // forward
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->weightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

//#ifdef CUDNN_HALF
    // backward delta
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dsrcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->ddstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->dweightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

    // forward
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->weightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

    // batch norm
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDescF16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
//#endif

    // batch norm
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));

    //printf("\n l->dilation = %d, l->pad = %d, l->size = %d \n", l->dilation, l->pad, l->size);
#if(CUDNN_MAJOR >= 6)
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l->convDesc, l->pad * l->dilation, l->pad* l->dilation, l->stride, l->stride, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));    // cudnn >= 6.0
#else
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l->convDesc, l->pad * l->dilation, l->pad * l->dilation, l->stride, l->stride, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION));    // cudnn 5.1
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

    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            (cudnnConvolutionFwdPreference_t)forward_algo,
            0,
            &l->fw_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            (cudnnConvolutionBwdDataPreference_t)backward_algo,
            0,
            &l->bd_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            (cudnnConvolutionBwdFilterPreference_t)backward_filter,
            0,
            &l->bf_algo));

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
    }
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, convolutional_layer *share_layer)
{
    int total_batch = batch*steps;
    int i;
    convolutional_layer l = { (LAYER_TYPE)0 };
    l.type = CONVOLUTIONAL;

    if (xnor) groups = 1;   // disable groups for XNOR-net
    if (groups < 1) groups = 1;

    l.share_layer = share_layer;
    l.index = index;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch = batch;
    l.steps = steps;
    l.stride = stride;
    l.dilation = dilation;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    l.learning_rate_scale = 1;
    l.nweights = (c / groups) * n * size * size;

    if (l.share_layer) {
        l.weights = l.share_layer->weights;
        l.weight_updates = l.share_layer->weight_updates;

        l.biases = l.share_layer->biases;
        l.bias_updates = l.share_layer->bias_updates;
    }
    else {
        l.weights = (float*)calloc(l.nweights, sizeof(float));
        l.weight_updates = (float*)calloc(l.nweights, sizeof(float));

        l.biases = (float*)calloc(n, sizeof(float));
        l.bias_updates = (float*)calloc(n, sizeof(float));
    }

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/groups));
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1, 1);   // rand_normal();
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.activation = activation;

    l.output = (float*)calloc(total_batch*l.outputs, sizeof(float));
    l.delta  = (float*)calloc(total_batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
        l.cweights = (char*)calloc(l.nweights, sizeof(char));
        l.scales = (float*)calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
        l.binary_input = (float*)calloc(l.inputs * l.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = l.out_h*l.out_w;
        l.bit_align = src_align + (align - src_align % align);

        l.mean_arr = (float*)calloc(l.n, sizeof(float));

        const size_t new_c = l.c / 32;
        size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
        l.bin_re_packed_input = (uint32_t*)calloc(in_re_packed_input_size, sizeof(uint32_t));

        l.lda_align = 256;  // AVX2
        int k = l.size*l.size*l.c;
        size_t k_aligned = k + (l.lda_align - k%l.lda_align);
        size_t t_bit_input_size = k_aligned * l.bit_align / 8;
        l.t_bit_input = (char*)calloc(t_bit_input_size, sizeof(char));
    }

    if(batch_normalize){
        if (l.share_layer) {
            l.scales = l.share_layer->scales;
            l.scale_updates = l.share_layer->scale_updates;
            l.mean = l.share_layer->mean;
            l.variance = l.share_layer->variance;
            l.mean_delta = l.share_layer->mean_delta;
            l.variance_delta = l.share_layer->variance_delta;
            l.rolling_mean = l.share_layer->rolling_mean;
            l.rolling_variance = l.share_layer->rolling_variance;
        }
        else {
            l.scales = (float*)calloc(n, sizeof(float));
            l.scale_updates = (float*)calloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                l.scales[i] = 1;
            }

            l.mean = (float*)calloc(n, sizeof(float));
            l.variance = (float*)calloc(n, sizeof(float));

            l.mean_delta = (float*)calloc(n, sizeof(float));
            l.variance_delta = (float*)calloc(n, sizeof(float));

            l.rolling_mean = (float*)calloc(n, sizeof(float));
            l.rolling_variance = (float*)calloc(n, sizeof(float));
        }

        l.x = (float*)calloc(total_batch * l.outputs, sizeof(float));
        l.x_norm = (float*)calloc(total_batch * l.outputs, sizeof(float));
    }
    if(adam){
        l.adam = 1;
        l.m = (float*)calloc(l.nweights, sizeof(float));
        l.v = (float*)calloc(l.nweights, sizeof(float));
        l.bias_m = (float*)calloc(n, sizeof(float));
        l.scale_m = (float*)calloc(n, sizeof(float));
        l.bias_v = (float*)calloc(n, sizeof(float));
        l.scale_v = (float*)calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }
        if (l.share_layer) {
            l.weights_gpu = l.share_layer->weights_gpu;
            l.weight_updates_gpu = l.share_layer->weight_updates_gpu;
            l.weights_gpu16 = l.share_layer->weights_gpu16;
            l.weight_updates_gpu16 = l.share_layer->weight_updates_gpu16;
            l.biases_gpu = l.share_layer->biases_gpu;
            l.bias_updates_gpu = l.share_layer->bias_updates_gpu;
        }
        else {
            l.weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
#ifdef CUDNN_HALF
            l.weights_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
            l.weight_updates_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
#endif
            l.biases_gpu = cuda_make_array(l.biases, n);
            l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
        }

        l.output_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
        l.delta_gpu = cuda_make_array(l.delta, total_batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.mean_arr_gpu = cuda_make_array(0, l.n);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            if (l.share_layer) {
                l.scales_gpu = l.share_layer->scales_gpu;
                l.scale_updates_gpu = l.share_layer->scale_updates_gpu;
                l.mean_gpu = l.share_layer->mean_gpu;
                l.variance_gpu = l.share_layer->variance_gpu;
                l.rolling_mean_gpu = l.share_layer->rolling_mean_gpu;
                l.rolling_variance_gpu = l.share_layer->rolling_variance_gpu;
                l.mean_delta_gpu = l.share_layer->mean_delta_gpu;
                l.variance_delta_gpu = l.share_layer->variance_delta_gpu;
            }
            else {
                l.scales_gpu = cuda_make_array(l.scales, n);
                l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

                l.mean_gpu = cuda_make_array(l.mean, n);
                l.variance_gpu = cuda_make_array(l.variance, n);

                l.rolling_mean_gpu = cuda_make_array(l.mean, n);
                l.rolling_variance_gpu = cuda_make_array(l.variance, n);

                l.mean_delta_gpu = cuda_make_array(l.mean, n);
                l.variance_delta_gpu = cuda_make_array(l.variance, n);
            }
            l.x_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
        }
#ifdef CUDNN
        create_convolutional_cudnn_tensors(&l);
        cudnn_convolutional_setup(&l, cudnn_fastest);
#endif
    }
#endif
    l.workspace_size = get_convolutional_workspace_size(l);

    //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else fprintf(stderr, "conv  ");
    fprintf(stderr, "%5d %2d x%2d /%2d(%d)%4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", n, size, size, stride, dilation, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.nweights; ++j){
            l.weights[i*l.nweights + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 1, 5, 5, 3, 2, 1, 5, 2, 1, 1, LEAKY, 1, 0, 0, 0, 0, 0, NULL);
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
    int total_batch = l->batch*l->steps;
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

    l->output = (float*)realloc(l->output, total_batch * l->outputs * sizeof(float));
    l->delta = (float*)realloc(l->delta, total_batch * l->outputs * sizeof(float));
    if(l->batch_normalize){
        l->x = (float*)realloc(l->x, total_batch * l->outputs * sizeof(float));
        l->x_norm = (float*)realloc(l->x_norm, total_batch * l->outputs * sizeof(float));
    }

    if (l->xnor) {
        //l->binary_input = realloc(l->inputs*l->batch, sizeof(float));
    }

#ifdef GPU
    if (old_w < w || old_h < h) {
        cuda_free(l->delta_gpu);
        cuda_free(l->output_gpu);

        l->delta_gpu = cuda_make_array(l->delta, total_batch*l->outputs);
        l->output_gpu = cuda_make_array(l->output, total_batch*l->outputs);

        if (l->batch_normalize) {
            cuda_free(l->x_gpu);
            cuda_free(l->x_norm_gpu);

            l->x_gpu = cuda_make_array(l->output, total_batch*l->outputs);
            l->x_norm_gpu = cuda_make_array(l->output, total_batch*l->outputs);
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
    l->workspace_size = get_convolutional_workspace_size(*l);

#ifdef CUDNN
    // check for excessive memory consumption
    size_t free_byte;
    size_t total_byte;
    CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
    if (l->workspace_size > free_byte || l->workspace_size >= total_byte / 2) {
        printf(" used slow CUDNN algo without Workspace! Need memory: %zu, available: %zu\n", l->workspace_size, (free_byte < total_byte/2) ? free_byte : total_byte/2);
        cudnn_convolutional_setup(l, cudnn_smallest);
        l->workspace_size = get_convolutional_workspace_size(*l);
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
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
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
    size_t i;

    for (i = 0; i < size; ++i) {
        float mean_val = 1;
        if(mean_arr != NULL) mean_val = fabs(mean_arr[i / (size / filters)]);
        if(get_bit(src, i)) dst[i] = mean_val;
        else dst[i] = -mean_val;
    }
}

void binary_align_weights(convolutional_layer *l)
{
    int m = l->n;   // (l->n / l->groups)
    int k = l->size*l->size*l->c;   // ->size*l->size*(l->c / l->groups)
    size_t new_lda = k + (l->lda_align - k % l->lda_align); // (k / 8 + 1) * 8;
    l->new_lda = new_lda;

    binarize_weights(l->weights, m, k, l->binary_weights);

    size_t align_weights_size = new_lda * m;
    l->align_bit_weights_size = align_weights_size / 8 + 1;
    float* align_weights = (float*)calloc(align_weights_size, sizeof(float));
    l->align_bit_weights = (char*)calloc(l->align_bit_weights_size, sizeof(char));

    size_t i, j;
    // align A without transpose
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = l->binary_weights[i*k + j];
        }
    }


    if (l->c % 32 == 0)
    //if(gpu_index < 0 && l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
    //if (l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
    {
        int fil, chan;
        const int items_per_filter = l->c * l->size * l->size;
        //const int dst_items_per_filter = new_lda;
        for (fil = 0; fil < l->n; ++fil)
        {
            for (chan = 0; chan < l->c; chan += 32)
            {
                const int items_per_channel = l->size*l->size;
                for (i = 0; i < items_per_channel; ++i)
                {
                    //uint32_t val = 0;
                    int c_pack;
                    for (c_pack = 0; c_pack < 32; ++c_pack) {
                        float src = l->binary_weights[fil*items_per_filter + (chan + c_pack)*items_per_channel + i];

                        //align_weights[fil*items_per_filter + chan*items_per_channel + i * 32 + c_pack] = src;

                        align_weights[fil*new_lda + chan*items_per_channel + i*32 + c_pack] = src;
                        //val |= (src << c);
                    }

                }
            }
        }

        //printf("\n l.index = %d \t aw[0] = %f, aw[1] = %f, aw[2] = %f, aw[3] = %f \n", l->index, align_weights[0], align_weights[1], align_weights[2], align_weights[3]);
        //memcpy(l->binary_weights, align_weights, (l->size * l->size * l->c * l->n) * sizeof(float));

        float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

        //if (l->n >= 32)
        if(gpu_index >= 0)
        {
            //int M = l->n;
            //int N = l->out_w*l->out_h;
            //printf("\n M = %d, N = %d, M %% 8 = %d, N %% 8 = %d - weights \n", M, N, M % 8, N % 8);
            //printf("\n l.w = %d, l.c = %d, l.n = %d \n", l->w, l->c, l->n);
            for (i = 0; i < align_weights_size / 8; ++i) l->align_bit_weights[i] = ~(l->align_bit_weights[i]);
        }



        get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
        //get_mean_array(l->binary_weights, m*new_lda, l->n, l->mean_arr);
    }
    else {
        float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

        get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
    }

    //l->mean_arr = calloc(l->n, sizeof(float));

    //get_mean_array(align_weights, align_weights_size, l->n, l->mean_arr);




#ifdef GPU
    cudaError_t status;
    l->align_workspace_size = l->bit_align * l->size * l->size * l->c;
    status = cudaMalloc((void **)&l->align_workspace_gpu, l->align_workspace_size * sizeof(float));
    status = cudaMalloc((void **)&l->transposed_align_workspace_gpu, l->align_workspace_size * sizeof(float));
    CHECK_CUDA(status);

    //l->align_bit_weights_gpu = cuda_make_array(l->align_bit_weights, l->align_bit_weights_size * sizeof(char)/sizeof(float));
    status = cudaMalloc((void **)&l->align_bit_weights_gpu, l->align_bit_weights_size);
    CHECK_CUDA(status);
    status = cudaMemcpy(l->align_bit_weights_gpu, l->align_bit_weights, l->align_bit_weights_size, cudaMemcpyHostToDevice);
    CHECK_CUDA(status);
    status = cudaMemcpy(l->binary_weights_gpu, l->binary_weights, m*k * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA(status);

    //l->mean_arr_gpu = cuda_make_array(l->mean_arr, l->n);
    cuda_push_array(l->mean_arr_gpu, l->mean_arr, l->n);
    CHECK_CUDA(cudaDeviceSynchronize());
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

    memset(*t_bit_input, 0, t_bit_input_size * sizeof(char));
    //int src_size = k * bit_align;

    // b - [bit_align, k] - [l.bit_align, l.size*l.size*l.c] = src_size
    // t_input - [bit_align, k] - [n', k]
    // t_bit_input - [new_ldb, n] - [k', n]

    //transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    transpose_bin((uint32_t*)b, (uint32_t*)*t_bit_input, k, n, bit_align, new_ldb, 8);

    return t_intput_size;
}


void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if (l.xnor && (!l.align_bit_weights || state.train)) {
        if (!l.align_bit_weights || state.train) {
            binarize_weights(l.weights, l.n, l.nweights, l.binary_weights);
            //printf("\n binarize_weights l.align_bit_weights = %p \n", l.align_bit_weights);
        }
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
    }

    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    int n = out_h*out_w;

    static int u = 0;
    u++;

    for(i = 0; i < l.batch; ++i){
        for (j = 0; j < l.groups; ++j) {

            float *a = l.weights + j*l.nweights / l.groups;
            float *b = state.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;

            //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
            //gemm_nn_custom(m, n, k, 1, a, k, b, n, c, n);
            if (l.xnor && l.align_bit_weights && !state.train)
            {
                memset(b, 0, l.bit_align*l.size*l.size*l.c * sizeof(float));

                if (l.c % 32 == 0)
                {
                    //printf(" l.index = %d - new XNOR \n", l.index);

                    int ldb_align = l.lda_align;
                    size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
                    //size_t t_intput_size = new_ldb * l.bit_align;// n;
                    //size_t t_bit_input_size = t_intput_size / 8;// +1;

                    int re_packed_input_size = l.c * l.w * l.h;
                    memset(state.workspace, 0, re_packed_input_size * sizeof(float));

                    const size_t new_c = l.c / 32;
                    size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
                    memset(l.bin_re_packed_input, 0, in_re_packed_input_size * sizeof(uint32_t));

                    //float *re_packed_input = calloc(l.c * l.w * l.h, sizeof(float));
                    //uint32_t *bin_re_packed_input = calloc(new_c * l.w * l.h + 1, sizeof(uint32_t));

                    // float32x4 by channel (as in cuDNN)
                    repack_input(state.input, state.workspace, l.w, l.h, l.c);

                    // 32 x floats -> 1 x uint32_t
                    float_to_bit(state.workspace, (unsigned char *)l.bin_re_packed_input, l.c * l.w * l.h);

                    //free(re_packed_input);

                    // slow - convolution the packed inputs and weights: float x 32 by channel (as in cuDNN)
                    //convolution_repacked((uint32_t *)bin_re_packed_input, (uint32_t *)l.align_bit_weights, l.output,
                    //    l.w, l.h, l.c, l.n, l.size, l.pad, l.new_lda, l.mean_arr);

                    // // then exit from if()


                    im2col_cpu_custom((float *)l.bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
                    //im2col_cpu((float *)bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, b);

                    //free(bin_re_packed_input);

                    int new_k = l.size*l.size*l.c / 32;

                    // good for (l.c == 64)
                    //gemm_nn_bin_32bit_packed(m, n, new_k, 1,
                    //    l.align_bit_weights, l.new_lda/32,
                    //    b, n,
                    //    c, n, l.mean_arr);

    // // then exit from if()

                    transpose_uint32((uint32_t *)state.workspace, (uint32_t*)l.t_bit_input, new_k, n, n, new_ldb);

                    // the main GEMM function
                    gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)l.align_bit_weights, new_ldb, (unsigned char*)l.t_bit_input, new_ldb, c, n, l.mean_arr);

                    // // alternative GEMM
                    //gemm_nn_bin_transposed_32bit_packed(m, n, new_k, 1,
                    //    l.align_bit_weights, l.new_lda/32,
                    //    t_bit_input, new_ldb / 32,
                    //    c, n, l.mean_arr);

                    //free(t_bit_input);

                }
                else
                { // else (l.c % 32 != 0)

                    //--------------------------------------------------------
                    //printf(" l.index = %d - old XNOR \n", l.index);

                    //im2col_cpu_custom_align(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, l.bit_align);
                    im2col_cpu_custom_bin(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, state.workspace, l.bit_align);

                    //size_t output_size = l.outputs;
                    //float *count_output = calloc(output_size, sizeof(float));
                    //size_t bit_output_size = output_size / 8 + 1;
                    //char *bit_output = calloc(bit_output_size, sizeof(char));

                    //size_t intput_size = n * k; // (out_h*out_w) X (l.size*l.size*l.c) : after im2col()
                    //size_t bit_input_size = intput_size / 8 + 1;
                    //char *bit_input = calloc(bit_input_size, sizeof(char));

                    //size_t weights_size = k * m; //l.size*l.size*l.c*l.n;
                    //size_t bit_weights_size = weights_size / 8 + 1;

                    //char *bit_weights = calloc(bit_weights_size, sizeof(char));
                    //float *mean_arr = calloc(l.n, sizeof(float));

                    // transpose B from NxK to KxN (x-axis (ldb = l.size*l.size*l.c) - should be multiple of 8 bits)
                    {
                        //size_t ldb_align = 256; // 256 bit for AVX2
                        int ldb_align = l.lda_align;
                        size_t new_ldb = k + (ldb_align - k%ldb_align);
                        size_t t_intput_size = binary_transpose_align_input(k, n, state.workspace, &l.t_bit_input, ldb_align, l.bit_align);

                        // 5x times faster than gemm()-float32
                        gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)l.align_bit_weights, new_ldb, (unsigned char*)l.t_bit_input, new_ldb, c, n, l.mean_arr);

                        //gemm_nn_custom_bin_mean_transposed(m, n, k, 1, bit_weights, k, t_bit_input, new_ldb, c, n, mean_arr);

                        //free(t_input);
                        //free(t_bit_input);
                        //}
                    }

                }

                add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

                //activate_array(l.output, m*n*l.batch, l.activation);
                activate_array_cpu_custom(l.output, m*n*l.batch, l.activation);
                return;

            }
            else {
                //printf(" l.index = %d - FP32 \n", l.index);
                float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;
                //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);

                im2col_cpu_ext(im,   // input
                    l.c / l.groups,     // input channels
                    l.h, l.w,           // input size (h, w)
                    l.size, l.size,     // kernel size (h, w)
                    l.pad, l.pad,       // padding (h, w)
                    l.stride, l.stride, // stride (h, w)
                    l.dilation, l.dilation, // dilation (h, w)
                    b);                 // output

                gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                // bit-count to float
            }
            c += n*m;
            state.input += l.c*l.h*l.w;
        }
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

    //activate_array(l.output, m*n*l.batch, l.activation);
    activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);

    if(l.binary || l.xnor) swap_binary(&l);
}


void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i, j;
    int m = l.n / l.groups;
    int n = l.size*l.size*l.c / l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if (l.batch_normalize) {
        backward_batchnorm_layer(l, state);
    }
    else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for (i = 0; i < l.batch; ++i) {
        for (j = 0; j < l.groups; ++j) {
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = state.workspace;
            float *c = l.weight_updates + j*l.nweights / l.groups;

            float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

            //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            im2col_cpu_ext(
                im,                 // input
                l.c / l.groups,     // input channels
                l.h, l.w,           // input size (h, w)
                l.size, l.size,     // kernel size (h, w)
                l.pad, l.pad,       // padding (h, w)
                l.stride, l.stride, // stride (h, w)
                l.dilation, l.dilation, // dilation (h, w)
                b);                 // output

            gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

            if (state.delta) {
                a = l.weights + j*l.nweights / l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = state.workspace;

                gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

                //col2im_cpu(state.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride,
                //     l.pad, state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w);

                col2im_cpu_ext(
                    state.workspace,        // input
                    l.c / l.groups,         // input channels (h, w)
                    l.h, l.w,               // input size (h, w)
                    l.size, l.size,         // kernel size (h, w)
                    l.pad, l.pad,           // padding (h, w)
                    l.stride, l.stride,     // stride (h, w)
                    l.dilation, l.dilation, // dilation (h, w)
                    state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w); // output (delta)
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    //int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if (l.scales) {
        axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}



image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c / l.groups;
    return float_to_image(w, h, c, l.weights + i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for (i = 0; i < l.n; ++i) {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for (i = 0; i < l.n; ++i) {
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
    image *weights = (image *)calloc(l.n, sizeof(image));
    int i;
    for (i = 0; i < l.n; ++i) {
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
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