#include "maxpool_layer.h"
#include "cuda.h"
#include "gemm.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}


void cudnn_maxpool_setup(layer *l)
{
#ifdef CUDNN
    cudnnStatus_t maxpool_status;
    maxpool_status = cudnnCreatePoolingDescriptor(&l->poolingDesc);

    maxpool_status = cudnnSetPooling2dDescriptor(
        l->poolingDesc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,    // CUDNN_PROPAGATE_NAN, CUDNN_NOT_PROPAGATE_NAN
        l->size,
        l->size,
        0, //l.pad,
        0, //l.pad,
        l->stride,
        l->stride);

    cudnnCreateTensorDescriptor(&l->srcTensorDesc);
    cudnnCreateTensorDescriptor(&l->dstTensorDesc);
    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
#endif // CUDNN
}


maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = { (LAYER_TYPE)0 };
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size) / stride + 1;
    l.out_h = (h + padding - size) / stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = (int*)calloc(output_size, sizeof(int));
    l.output = (float*)calloc(output_size, sizeof(float));
    l.delta = (float*)calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);

    cudnn_maxpool_setup(&l);

    #endif  // GPU
	l.bflops = (l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size) / l->stride + 1;
    l->out_h = (h + l->pad - l->size) / l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = (int*)realloc(l->indexes, output_size * sizeof(int));
    l->output = (float*)realloc(l->output, output_size * sizeof(float));
    l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

#ifdef GPU
    CHECK_CUDA(cudaFree((float *)l->indexes_gpu));
    CHECK_CUDA(cudaFree(l->output_gpu));
    CHECK_CUDA(cudaFree(l->delta_gpu));
    l->indexes_gpu = cuda_make_int_array(output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);

    cudnn_maxpool_setup(l);
#endif
}

void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    if (!state.train) {
        forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
        return;
    }

    int b,i,j,k,m,n;
    int w_offset = -l.pad / 2;
    int h_offset = -l.pad / 2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}
