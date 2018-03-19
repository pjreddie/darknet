#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")  
#endif

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabs(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabs(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void cuda_f32_to_f16(float* input_f32, size_t size, half *output_f16)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f16[idx] = __float2half(input_f32[idx]);
	//if (idx < size) *((unsigned short *)output_f16 + idx) = __float2half(input_f32[idx]);
}

void cuda_convert_f32_to_f16(float* input_f32, size_t size, float *output_f16) {
	cuda_f32_to_f16 <<< size / BLOCK + 1, BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (half *)output_f16);
}

__global__ void cuda_f16_to_f32(half* input_f16, size_t size, float *output_f32)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f32[idx] = __half2float(input_f16[idx]);
	//if (idx < size) output_f32[idx] = __half2float(*((unsigned short *)input_f16 + idx));
}

void cuda_convert_f16_to_f32(float* input_f16, size_t size, float *output_f32) {
	cuda_f16_to_f32 <<< size / BLOCK + 1, BLOCK, 0, get_cuda_stream() >>> ((half *)input_f16, size, output_f32);
}

half *cuda_make_f16_from_f32_array(float *src, size_t n)
{
	half *dst16;
	size_t size = sizeof(half)*n;
	check_error(cudaMalloc((void **)&dst16, size));
	if (src) {
		cuda_convert_f32_to_f16(src, n, (float *)dst16);
	}
	if (!dst16) error("Cuda malloc failed\n");
	return dst16;
}

void forward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        state.input = l.binary_input_gpu;
    }

#ifdef CUDNN
	float one = 1;	// alpha[0], beta[0] is float for HALF and FLOAT
	float alpha = 1, beta = 0; 

#ifdef CUDNN_HALF
	// Note: For improved performance it is advised to use beta[0] = 0.0. 
	// For Tensor Core: cudnnSetConvolutionMathType() where cudnnMathType_t mathType = CUDNN_TENSOR_OP_MATH;
	// 1. or CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM and use CUDNN_DATA_HALF
	// 2. or CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
	// More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops

	const size_t input16_size = l.batch*l.c*l.w*l.h;
	const size_t output16_size = l.batch*l.out_c*l.out_h*l.out_w;

	if (*state.net.max_input16_size < input16_size) {
		//printf("\n input16_size: cur = %zu \t max = %zu \n", input16_size, *state.net.max_input16_size);
		*state.net.max_input16_size = input16_size;
		if (*state.net.input16_gpu) cuda_free(*state.net.input16_gpu);
		*state.net.input16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_input16_size);
	}
	float *input16 = *state.net.input16_gpu;

	if (*state.net.max_output16_size < output16_size) {
		*state.net.max_output16_size = output16_size;
		if (*state.net.output16_gpu) cuda_free(*state.net.output16_gpu);
		*state.net.output16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_output16_size);
	}
	float *output16 = *state.net.output16_gpu;

	cuda_convert_f32_to_f16(state.input, input16_size, input16);

	//fill_ongpu(output16_size / 2, 0, (float *)output16, 1);
	cudnnConvolutionForward(cudnn_handle(),
		&alpha,
		l.srcTensorDesc,
		input16,
		l.weightDesc,
		l.weights_gpu16,
		l.convDesc,
		l.fw_algo,
		state.workspace,
		l.workspace_size,
		&beta,
		l.dstTensorDesc,
		output16);
	
	cuda_convert_f16_to_f32(output16, output16_size, l.output_gpu);

#else

    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                state.input,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);
#endif


#else
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        float * a = l.weights_gpu;
        float * b = state.workspace;
        float * c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
	}
	else {
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
	}

    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
	//cudaDeviceSynchronize();	// for correct profiling of performance
}

void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.x_gpu, 1, l.delta_gpu, 1);
    } else {
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.output_gpu, 1, l.delta_gpu, 1);
    }
    float *original_input = state.input;

    if(l.xnor) state.input = l.binary_input_gpu;
#ifdef CUDNN
	float one = 1;
	float alpha = 1, beta = 0;

#ifdef CUDNN_HALF
		
	const size_t input16_size = l.batch*l.c*l.w*l.h;
	const size_t delta16_size = l.batch*l.n*l.out_w*l.out_h;
	
	if (*state.net.max_input16_size < input16_size) {		
		*state.net.max_input16_size = input16_size;
		if(*state.net.input16_gpu) cuda_free(*state.net.input16_gpu);
		*state.net.input16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_input16_size);
	}
	float *input16 = *state.net.input16_gpu;

	if (*state.net.max_output16_size < delta16_size) {
		*state.net.max_output16_size = delta16_size;
		if(*state.net.output16_gpu) cuda_free(*state.net.output16_gpu);
		*state.net.output16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *state.net.max_output16_size);
	}
	float *delta16 = *state.net.output16_gpu;

	cuda_convert_f32_to_f16(state.input, input16_size, input16);
	cuda_convert_f32_to_f16(l.delta_gpu, delta16_size, delta16);
	
	// convert input: state.input (x), l.delta_gpu (y) from fp32 to fp16
	// get output: l.weight_updates_gpu (dw) and convert it to fp32 (ONLY if it is fp16)

	// calculate conv weight updates
	// Already: l.weight_updates_gpu = (l.weight_updates_gpu - l.weight*decay*batch*subdivision)*momentum
	//   so we should copy f32 to f16, or compute: f16=(w_up - w*d*b*s)*m
	cuda_convert_f32_to_f16(l.weight_updates_gpu, l.c*l.n*l.size*l.size, l.weight_updates_gpu16);

	cudnnConvolutionBackwardFilter(cudnn_handle(),
		&one,
		l.srcTensorDesc,
		input16, //state.input,
		l.ddstTensorDesc,
		delta16, //l.delta_gpu,
		l.convDesc,
		l.bf_algo,
		state.workspace,
		l.workspace_size,
		&one,
		l.dweightDesc,
		l.weight_updates_gpu16);	// l.weight_updates_gpu);

	cuda_convert_f16_to_f32(l.weight_updates_gpu16, l.c*l.n*l.size*l.size, l.weight_updates_gpu);

	if (state.delta) {
		if (l.binary || l.xnor) swap_binary(&l);

		// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
		// calculate delta for the next layer
		// convert input: l.weights_gpu (w), l.delta_gpu (dy) from fp32 to fp16
		// get output: state.delta (dx) and convert it to fp32 (ONLY if it is fp16)	
		cudnnConvolutionBackwardData(cudnn_handle(),
			&alpha,
			l.weightDesc,
			l.weights_gpu16, //l.weights_gpu,
			l.ddstTensorDesc,
			delta16, //l.delta_gpu,
			l.convDesc,
			l.bd_algo,
			state.workspace,
			l.workspace_size,
			&beta,
			l.dsrcTensorDesc,
			input16);	// state.delta);

		cuda_convert_f16_to_f32(input16, input16_size, state.delta);

		if (l.binary || l.xnor) swap_binary(&l);
		if (l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
	}
#else	// CUDNN_HALF

	// calculate conv weight updates
	// if used: beta=1 then loss decreases faster
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            state.input,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(state.delta){
        if(l.binary || l.xnor) swap_binary(&l);
		// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
		// calculate delta for the next layer
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                state.delta);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
    }

#endif	// CUDNN_HALF

#else	// CUDNN
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = state.workspace;
        float * c = l.weight_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            if(l.binary || l.xnor) swap_binary(&l);
            float * a = l.weights_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary(&l);
            }
            if(l.xnor) gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
#ifdef CUDNN_HALF
	cuda_convert_f32_to_f16(layer.weights_gpu, layer.c*layer.n*layer.size*layer.size, layer.weights_gpu16);
#endif
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    if(layer.scales_gpu){
        axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
        scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);
    }

    if(layer.adam){
        scal_ongpu(size, layer.B1, layer.m_gpu, 1);
        scal_ongpu(size, layer.B2, layer.v_gpu, 1);

        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);

        axpy_ongpu(size, -(1-layer.B1), layer.weight_updates_gpu, 1, layer.m_gpu, 1);
        mul_ongpu(size, layer.weight_updates_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, (1-layer.B2), layer.weight_updates_gpu, 1, layer.v_gpu, 1);

        adam_gpu(size, layer.weights_gpu, layer.m_gpu, layer.v_gpu, layer.B1, layer.B2, learning_rate/batch, layer.eps, layer.t+1);
        fill_ongpu(size, 0, layer.weight_updates_gpu, 1);
    }else{
		// update weights:
		// weights_gpu = weights_gpu*(1 - decay*lr) + weight_updates_gpu*lr / (batch*subdivision) =
		//  weights_gpu*(1 - 0.0005*0.001) + weight_updates_gpu*0.001/(64*8) = 
		//  weights_gpu * 0.999 999 5 + weight_updates_gpu * 0.000 001 953125
		// 
		// weight_updates_gpu = (weight_updates_gpu - weights_gpu*decay*batch*subdivision)*momentum = 
		//  (weight_updates_gpu - weights_gpu * 0.0005 * 64 * 8) * 0.9 = 
		//  weight_updates_gpu*0.9 - weights_gpu*0.2304
        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, learning_rate/batch, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
        scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);
    }
}


