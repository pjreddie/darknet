#include "maxpool_layer.h"
#include "convolutional_layer.h"
#include "dark_cuda.h"
#include "utils.h"
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

void create_maxpool_cudnn_tensors(layer *l)
{
#ifdef CUDNN
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&l->poolingDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->srcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dstTensorDesc));
#endif // CUDNN
}

void cudnn_maxpool_setup(layer *l)
{
#ifdef CUDNN
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(
        l->poolingDesc,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,    // CUDNN_PROPAGATE_NAN, CUDNN_NOT_PROPAGATE_NAN
        l->size,
        l->size,
        l->pad/2, //0, //l.pad,
        l->pad/2, //0, //l.pad,
        l->stride_x,
        l->stride_y));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));
#endif // CUDNN
}


void cudnn_local_avgpool_setup(layer *l)
{
#ifdef CUDNN
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(
        l->poolingDesc,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,    // CUDNN_PROPAGATE_NAN, CUDNN_NOT_PROPAGATE_NAN
        l->size,
        l->size,
        l->pad / 2, //0, //l.pad,
        l->pad / 2, //0, //l.pad,
        l->stride_x,
        l->stride_y));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));
#endif // CUDNN
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x, int stride_y, int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train)
{
    maxpool_layer l = { (LAYER_TYPE)0 };
    l.avgpool = avgpool;
    if (avgpool) l.type = LOCAL_AVGPOOL;
    else l.type = MAXPOOL;
    l.train = train;

    const int blur_stride_x = stride_x;
    const int blur_stride_y = stride_y;
    l.antialiasing = antialiasing;
    if (antialiasing) {
        stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
    }

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.maxpool_depth = maxpool_depth;
    l.out_channels = out_channels;
    if (maxpool_depth) {
        l.out_c = out_channels;
        l.out_w = l.w;
        l.out_h = l.h;
    }
    else {
        l.out_w = (w + padding - size) / stride_x + 1;
        l.out_h = (h + padding - size) / stride_y + 1;
        l.out_c = c;
    }
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride_x;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    int output_size = l.out_h * l.out_w * l.out_c * batch;

    if (train) {
        if (!avgpool) l.indexes = (int*)xcalloc(output_size, sizeof(int));
        l.delta = (float*)xcalloc(output_size, sizeof(float));
    }
    l.output = (float*)xcalloc(output_size, sizeof(float));
    if (avgpool) {
        l.forward = forward_local_avgpool_layer;
        l.backward = backward_local_avgpool_layer;
    }
    else {
        l.forward = forward_maxpool_layer;
        l.backward = backward_maxpool_layer;
    }
#ifdef GPU
    if (avgpool) {
        l.forward_gpu = forward_local_avgpool_layer_gpu;
        l.backward_gpu = backward_local_avgpool_layer_gpu;
    }
    else {
        l.forward_gpu = forward_maxpool_layer_gpu;
        l.backward_gpu = backward_maxpool_layer_gpu;
    }

    if (train) {
        if (!avgpool) l.indexes_gpu = cuda_make_int_array(output_size);
        l.delta_gpu = cuda_make_array(l.delta, output_size);
    }
    l.output_gpu  = cuda_make_array(l.output, output_size);
    create_maxpool_cudnn_tensors(&l);
    if (avgpool) cudnn_local_avgpool_setup(&l);
    else cudnn_maxpool_setup(&l);

#endif  // GPU
    l.bflops = (l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    if (avgpool) {
        if (stride_x == stride_y)
            fprintf(stderr, "avg               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
        else
            fprintf(stderr, "avg              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, stride_y, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    }
    else {
        if (maxpool_depth)
            fprintf(stderr, "max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
        else if (stride_x == stride_y)
            fprintf(stderr, "max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
        else
            fprintf(stderr, "max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, stride_y, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    }

    if (l.antialiasing) {
        printf("AA:  ");
        l.input_layer = (layer*)calloc(1, sizeof(layer));
        int blur_size = 3;
        int blur_pad = blur_size / 2;
        if (l.antialiasing == 2) {
            blur_size = 2;
            blur_pad = 0;
        }
        *(l.input_layer) = make_convolutional_layer(batch, 1, l.out_h, l.out_w, l.out_c, l.out_c, l.out_c, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, LINEAR, 0, 0, 0, 0, 0, 1, 0, NULL, 0, 0, train);
        const int blur_nweights = l.out_c * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
        int i;
        if (blur_size == 2) {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 4.f;
                l.input_layer->weights[i + 1] = 1 / 4.f;
                l.input_layer->weights[i + 2] = 1 / 4.f;
                l.input_layer->weights[i + 3] = 1 / 4.f;
            }
        }
        else {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 16.f;
                l.input_layer->weights[i + 1] = 2 / 16.f;
                l.input_layer->weights[i + 2] = 1 / 16.f;

                l.input_layer->weights[i + 3] = 2 / 16.f;
                l.input_layer->weights[i + 4] = 4 / 16.f;
                l.input_layer->weights[i + 5] = 2 / 16.f;

                l.input_layer->weights[i + 6] = 1 / 16.f;
                l.input_layer->weights[i + 7] = 2 / 16.f;
                l.input_layer->weights[i + 8] = 1 / 16.f;
            }
        }
        for (i = 0; i < l.out_c; ++i) l.input_layer->biases[i] = 0;
#ifdef GPU
        if (gpu_index >= 0) {
            if (l.antialiasing) l.input_antialiasing_gpu = cuda_make_array(NULL, l.batch*l.outputs);
            push_convolutional_layer(*(l.input_layer));
        }
#endif  // GPU
    }

    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size) / l->stride_x + 1;
    l->out_h = (h + l->pad - l->size) / l->stride_y + 1;
    l->outputs = l->out_w * l->out_h * l->out_c;
    int output_size = l->outputs * l->batch;

    if (l->train) {
        if (!l->avgpool) l->indexes = (int*)xrealloc(l->indexes, output_size * sizeof(int));
        l->delta = (float*)xrealloc(l->delta, output_size * sizeof(float));
    }
    l->output = (float*)xrealloc(l->output, output_size * sizeof(float));

#ifdef GPU
    CHECK_CUDA(cudaFree(l->output_gpu));
    l->output_gpu  = cuda_make_array(l->output, output_size);

    if (l->train) {
        if (!l->avgpool) {
            CHECK_CUDA(cudaFree((float *)l->indexes_gpu));
            l->indexes_gpu = cuda_make_int_array(output_size);
        }
        CHECK_CUDA(cudaFree(l->delta_gpu));
        l->delta_gpu = cuda_make_array(l->delta, output_size);
    }

    if(l->avgpool) cudnn_local_avgpool_setup(l);
    else cudnn_maxpool_setup(l);
#endif
}

void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    if (l.maxpool_depth)
    {
        int b, i, j, k, g;
        for (b = 0; b < l.batch; ++b) {
            #pragma omp parallel for
            for (i = 0; i < l.h; ++i) {
                for (j = 0; j < l.w; ++j) {
                    for (g = 0; g < l.out_c; ++g)
                    {
                        int out_index = j + l.w*(i + l.h*(g + l.out_c*b));
                        float max = -FLT_MAX;
                        int max_i = -1;

                        for (k = g; k < l.c; k += l.out_c)
                        {
                            int in_index = j + l.w*(i + l.h*(k + l.c*b));
                            float val = state.input[in_index];

                            max_i = (val > max) ? in_index : max_i;
                            max = (val > max) ? val : max;
                        }
                        l.output[out_index] = max;
                        if (l.indexes) l.indexes[out_index] = max_i;
                    }
                }
            }
        }
        return;
    }


    if (!state.train && l.stride_x == l.stride_y) {
        forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
    }
    else
    {

        int b, i, j, k, m, n;
        int w_offset = -l.pad / 2;
        int h_offset = -l.pad / 2;

        int h = l.out_h;
        int w = l.out_w;
        int c = l.c;

        for (b = 0; b < l.batch; ++b) {
            for (k = 0; k < c; ++k) {
                for (i = 0; i < h; ++i) {
                    for (j = 0; j < w; ++j) {
                        int out_index = j + w*(i + h*(k + c*b));
                        float max = -FLT_MAX;
                        int max_i = -1;
                        for (n = 0; n < l.size; ++n) {
                            for (m = 0; m < l.size; ++m) {
                                int cur_h = h_offset + i*l.stride_y + n;
                                int cur_w = w_offset + j*l.stride_x + m;
                                int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                                int valid = (cur_h >= 0 && cur_h < l.h &&
                                    cur_w >= 0 && cur_w < l.w);
                                float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                                max_i = (val > max) ? index : max_i;
                                max = (val > max) ? val : max;
                            }
                        }
                        l.output[out_index] = max;
                        if (l.indexes) l.indexes[out_index] = max_i;
                    }
                }
            }
        }
    }

    if (l.antialiasing) {
        network_state s = { 0 };
        s.train = state.train;
        s.workspace = state.workspace;
        s.net = state.net;
        s.input = l.output;
        forward_convolutional_layer(*(l.input_layer), s);
        //simple_copy_ongpu(l.outputs*l.batch, l.output, l.input_antialiasing);
        memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));
    }
}

void backward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    #pragma omp parallel for
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}


void forward_local_avgpool_layer(const maxpool_layer l, network_state state)
{
    int b, i, j, k, m, n;
    int w_offset = -l.pad / 2;
    int h_offset = -l.pad / 2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for (b = 0; b < l.batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                    int out_index = j + w*(i + h*(k + c*b));
                    float avg = 0;
                    int counter = 0;
                    for (n = 0; n < l.size; ++n) {
                        for (m = 0; m < l.size; ++m) {
                            int cur_h = h_offset + i*l.stride_y + n;
                            int cur_w = w_offset + j*l.stride_x + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                cur_w >= 0 && cur_w < l.w);
                            if (valid) {
                                counter++;
                                avg += state.input[index];
                            }

                        }
                    }
                    l.output[out_index] = avg / counter;
                }
            }
        }
    }
}

void backward_local_avgpool_layer(const maxpool_layer l, network_state state)
{

    int b, i, j, k, m, n;
    int w_offset = -l.pad / 2;
    int h_offset = -l.pad / 2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for (b = 0; b < l.batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                    int out_index = j + w*(i + h*(k + c*b));
                    for (n = 0; n < l.size; ++n) {
                        for (m = 0; m < l.size; ++m) {
                            int cur_h = h_offset + i*l.stride_y + n;
                            int cur_w = w_offset + j*l.stride_x + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                cur_w >= 0 && cur_w < l.w);

                            if (valid) state.delta[index] += l.delta[out_index] / (l.size*l.size);
                        }
                    }

                }
            }
        }
    }

}
