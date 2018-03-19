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
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
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
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l, int cudnn_preference)
{

#ifdef CUDNN_HALF
	// TRUE_HALF_CONFIG is only supported on architectures with true fp16 support (compute capability 5.3 and 6.0): 
	//   Tegra X1, Jetson TX1, DRIVE CX, DRIVE PX, Quadro GP100, Tesla P100
	// PSEUDO_HALF_CONFIG is required for Tensor Cores - our case!
	const cudnnDataType_t data_type = CUDNN_DATA_HALF;
#else
	cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#endif

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

	// batch norm
	cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);
#if(CUDNN_MAJOR >= 6)
	cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);	// cudnn >= 6.0
#else
	cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);	// cudnn 5.1
#endif
	int forward_algo = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	int backward_algo = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
	int backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
	if (cudnn_preference == cudnn_smallest) 
	{
		forward_algo = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
		backward_algo = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
		backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
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
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
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
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l, cudnn_fastest);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

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
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
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
		printf(" used slow CUDNN algo without Workspace! \n");
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

void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
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

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(state.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        state.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
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

