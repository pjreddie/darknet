#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer layer = { (LAYER_TYPE)0 };
    layer.type = BATCHNORM;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.output = (float*)calloc(h * w * c * batch, sizeof(float));
    layer.delta = (float*)calloc(h * w * c * batch, sizeof(float));
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    layer.scales = (float*)calloc(c, sizeof(float));
    layer.scale_updates = (float*)calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        layer.scales[i] = 1;
    }

    layer.mean = (float*)calloc(c, sizeof(float));
    layer.variance = (float*)calloc(c, sizeof(float));

    layer.rolling_mean = (float*)calloc(c, sizeof(float));
    layer.rolling_variance = (float*)calloc(c, sizeof(float));

    layer.forward = forward_batchnorm_layer;
    layer.backward = backward_batchnorm_layer;
#ifdef GPU
    layer.forward_gpu = forward_batchnorm_layer_gpu;
    layer.backward_gpu = backward_batchnorm_layer_gpu;

    layer.output_gpu =  cuda_make_array(layer.output, h * w * c * batch);
    layer.delta_gpu =   cuda_make_array(layer.delta, h * w * c * batch);

    layer.scales_gpu = cuda_make_array(layer.scales, c);
    layer.scale_updates_gpu = cuda_make_array(layer.scale_updates, c);

    layer.mean_gpu = cuda_make_array(layer.mean, c);
    layer.variance_gpu = cuda_make_array(layer.variance, c);

    layer.rolling_mean_gpu = cuda_make_array(layer.mean, c);
    layer.rolling_variance_gpu = cuda_make_array(layer.variance, c);

    layer.mean_delta_gpu = cuda_make_array(layer.mean, c);
    layer.variance_delta_gpu = cuda_make_array(layer.variance, c);

    layer.x_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
    layer.x_norm_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
#ifdef CUDNN
    cudnnCreateTensorDescriptor(&layer.normTensorDesc);
    cudnnCreateTensorDescriptor(&layer.normDstTensorDesc);
    cudnnSetTensor4dDescriptor(layer.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, layer.batch, layer.out_c, layer.out_h, layer.out_w);
    cudnnSetTensor4dDescriptor(layer.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, layer.out_c, 1, 1);
#endif
#endif
    return layer;
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
    fprintf(stderr, "Not implemented\n");
}

void forward_batchnorm_layer(layer l, network_state state)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_h = l.out_w = 1;
    }
    if(state.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu(l.out_c, .9, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .1, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .9, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .1, l.variance, 1, l.rolling_variance, 1);

        copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(const layer l, network_state state)
{
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network_state state)
{
    if (l.type == BATCHNORM) simple_copy_ongpu(l.outputs*l.batch, state.input, l.output_gpu);
        //copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);

    if (state.train) {
        simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.x_gpu);
        //copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            l.normDstTensorDesc,
            l.x_gpu,                // input
            l.normDstTensorDesc,
            l.output_gpu,            // output
            l.normTensorDesc,
            l.scales_gpu,
            l.biases_gpu,
            .01,
            l.rolling_mean_gpu,        // output (should be FP32)
            l.rolling_variance_gpu,    // output (should be FP32)
            .00001,
            l.mean_gpu,            // output (should be FP32)
            l.variance_gpu);    // output (should be FP32)

        if (state.net.try_fix_nan) {
            fix_nan_and_inf(l.scales_gpu, l.n);
            fix_nan_and_inf(l.biases_gpu, l.n);
            fix_nan_and_inf(l.mean_gpu, l.n);
            fix_nan_and_inf(l.variance_gpu, l.n);
            fix_nan_and_inf(l.rolling_mean_gpu, l.n);
            fix_nan_and_inf(l.rolling_variance_gpu, l.n);
        }
#else
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_ongpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_ongpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif
    }
    else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }

}

void backward_batchnorm_layer_gpu(layer l, network_state state)
{
    if (!state.train) {
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL,
        &one,
        &zero,
        &one,
        &one,
        l.normDstTensorDesc,
        l.x_gpu,                // input
        l.normDstTensorDesc,
        l.delta_gpu,            // input
        l.normDstTensorDesc,
        l.x_norm_gpu,            // output
        l.normTensorDesc,
        l.scales_gpu,            // output (should be FP32)
        l.scale_updates_gpu,    // output (should be FP32)
        l.bias_updates_gpu,        // output (should be FP32)
        .00001,
        l.mean_gpu,                // input (should be FP32)
        l.variance_gpu);        // input (should be FP32)
    simple_copy_ongpu(l.outputs*l.batch, l.x_norm_gpu, l.delta_gpu);
    //copy_ongpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif
    if (l.type == BATCHNORM) simple_copy_ongpu(l.outputs*l.batch, l.delta_gpu, state.delta);
        //copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);

    if (state.net.try_fix_nan) {
        fix_nan_and_inf(l.scale_updates_gpu, l.n);
        fix_nan_and_inf(l.bias_updates_gpu, l.n);
    }
}
#endif
