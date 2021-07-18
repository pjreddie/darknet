#include "dark_cuda.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "gru_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"

//#ifdef OPENCV
//#include <opencv2/highgui/highgui_c.h>
//#endif

#include "http_stream.h"

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

typedef struct time_benchmark_layers {
    float time;
    int layer_id, layer_type;
} time_benchmark_layers;

int time_comparator(const void *pa, const void *pb)
{
    time_benchmark_layers a = *(time_benchmark_layers *)pa;
    time_benchmark_layers b = *(time_benchmark_layers *)pb;
    float diff = a.time - b.time;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void forward_network_gpu(network net, network_state state)
{
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    //printf("\n");
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu && state.train){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

        l.forward_gpu(l, state);

        if (net.benchmark_layers) {
            CHECK_CUDA(cudaDeviceSynchronize());
            end_time = get_time_point();
            const double took_time = (end_time - start_time) / 1000;
            const double alpha = 0.9;
            if (avg_time_per_layer[i].time == 0) {
                avg_time_per_layer[i].layer_id = i;
                avg_time_per_layer[i].layer_type = l.type;
                avg_time_per_layer[i].time = took_time;
            }
            else avg_time_per_layer[i].time = avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

            sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
            printf("\n fw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i, l.type, took_time, avg_time_per_layer[i].time);
        }

        if(net.wait_stream)
            cudaStreamSynchronize(get_cuda_stream());
        state.input = l.output_gpu;
        //cudaDeviceSynchronize();

        /*
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
        cudaStreamSynchronize(get_cuda_stream());
        float avg_val = 0;
        int k;
        for (k = 0; k < l.outputs; ++k) avg_val += l.output[k];
        printf(" i: %d - avg_val = %f \n", i, avg_val / l.outputs);
        */

/*
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        if (l.out_w >= 0 && l.out_h >= 1 && l.c >= 3) {
            int j;
            for (j = 0; j < l.out_c; ++j) {
                image img = make_image(l.out_w, l.out_h, 3);
                memcpy(img.data, l.output + l.out_w*l.out_h*j, l.out_w*l.out_h * 1 * sizeof(float));
                memcpy(img.data + l.out_w*l.out_h * 1, l.output + l.out_w*l.out_h*j, l.out_w*l.out_h * 1 * sizeof(float));
                memcpy(img.data + l.out_w*l.out_h * 2, l.output + l.out_w*l.out_h*j, l.out_w*l.out_h * 1 * sizeof(float));
                char buff[256];
                sprintf(buff, "layer-%d slice-%d", i, j);
                show_image(img, buff);
                save_image(img, buff);
            }
            cvWaitKey(0); // wait press-key in console
            cvDestroyAllWindows();
        }
*/
    }

    if (net.benchmark_layers) {
        printf("\n\nSorted by time (forward):\n");
        qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);
        for (i = 0; i < net.n; ++i) {
            //printf("layer %d - type: %d - avg_time %lf ms \n", avg_time_per_layer[i].layer_id, avg_time_per_layer[i].layer_type, avg_time_per_layer[i].time);
            printf("%d - fw-sort-layer %d - type: %d - avg_time %lf ms \n", i, sorted_avg_time_per_layer[i].layer_id, sorted_avg_time_per_layer[i].layer_type, sorted_avg_time_per_layer[i].time);
        }
    }

    //cudaStreamSynchronize(get_cuda_stream());   // sync CUDA-functions
    //cudaDeviceSynchronize();
}

void backward_network_gpu(network net, network_state state)
{
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    state.workspace = net.workspace;
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if (l.stopbackward == 1) break;
        if (l.stopbackward > get_current_iteration(net)) break;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
            if (net.optimized_memory && !prev.keep_delta_gpu) {
                state.delta = net.state_delta_gpu;
            }
        }
        if (l.onlyforward) continue;

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

        l.backward_gpu(l, state);

        if (net.benchmark_layers) {
            CHECK_CUDA(cudaDeviceSynchronize());
            end_time = get_time_point();
            const double took_time = (end_time - start_time) / 1000;
            const double alpha = 0.9;
            if (avg_time_per_layer[i].time == 0) {
                avg_time_per_layer[i].layer_id = i;
                avg_time_per_layer[i].layer_type = l.type;
                avg_time_per_layer[i].time = took_time;
            }
            else avg_time_per_layer[i].time = avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

            sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
            printf("\n bw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i, l.type, took_time, avg_time_per_layer[i].time);
        }

        if (i != 0) {
            layer prev = net.layers[i - 1];
            if (net.optimized_memory && state.delta && !prev.keep_delta_gpu) {
                if (prev.delta_gpu != state.delta) simple_copy_ongpu(prev.outputs*prev.batch, state.delta, prev.delta_gpu);
                fill_ongpu(prev.outputs*prev.batch, 0, net.state_delta_gpu, 1);
            }
        }

        /*
        if(i != 0)
        {
            layer l = net.layers[i - 1];
            int state_delta_nan_inf = is_nan_or_inf(state.delta, l.outputs * l.batch);
            int state_input_nan_inf = is_nan_or_inf(state.input, l.outputs * l.batch);
            printf("\n i - %d  is_nan_or_inf(s.delta) = %d \n", i, state_delta_nan_inf);
            printf(" i - %d  is_nan_or_inf(s.input) = %d \n", i, state_input_nan_inf);
            if (state_delta_nan_inf || state_input_nan_inf) { printf(" found "); getchar(); }
        }
        */
    }

    if (net.adversarial && net.attention)
    {
        int img_size = net.w * net.h * net.c;
        float *original_input_cpu = (float *)xcalloc(img_size, sizeof(float));
        float *original_delta_cpu = (float *)xcalloc(img_size, sizeof(float));
        cuda_pull_array(original_input, original_input_cpu, img_size);
        cuda_pull_array(original_delta, original_delta_cpu, img_size);

        image attention_img = make_attention_image(img_size, original_delta_cpu, original_input_cpu, net.w, net.h, net.c, 0.7);
        show_image(attention_img, "attention_img");
        resize_window_cv("attention_img", 500, 500);

        //static int img_counter = 0;
        //img_counter++;
        //char buff[256];
        //sprintf(buff, "attention_img_%d.png", img_counter);
        //save_image_png(attention_img, buff);
        free_image(attention_img);

        image attention_mask_img = make_attention_image(img_size, original_delta_cpu, original_delta_cpu, net.w, net.h, net.c, 1.0);
        show_image(attention_mask_img, "attention_mask_img");
        resize_window_cv("attention_mask_img", 500, 500);

        //sprintf(buff, "attention_mask_img_%d.png", img_counter);
        //save_image_png(attention_mask_img, buff);
        free_image(attention_mask_img);

        free(original_input_cpu);
        free(original_delta_cpu);
    }
    if (net.adversarial) {
        int x_size = get_network_input_size(net)*net.batch;
        printf(" x_size = %d, original_delta = %p, original_input = %p, net.learning_rate = %f \n",
            x_size, original_delta, original_input, net.learning_rate);
        axpy_ongpu(x_size, net.learning_rate, original_delta, 1, original_input, 1);
        constrain_min_max_ongpu(x_size, 0, 1, original_input, 1);
    }

    if (net.benchmark_layers) {
        printf("\n\nSorted by time (backward):\n");
        qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);
        for (i = 0; i < net.n; ++i) {
            //printf("layer %d - type: %d - avg_time %lf ms \n", avg_time_per_layer[i].layer_id, avg_time_per_layer[i].layer_type, avg_time_per_layer[i].time);
            printf("%d - bw-sort-layer %d - type: %d - avg_time %lf ms \n", i, sorted_avg_time_per_layer[i].layer_id, sorted_avg_time_per_layer[i].layer_type, sorted_avg_time_per_layer[i].time);
        }
    }
}

void update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    const int iteration_num = (*net.seen) / (net.batch * net.subdivisions);
    int i;
    int update_batch = net.batch*net.subdivisions * get_sequence_value(net);
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if (l.train == 0) continue;

        l.t = get_current_batch(net);
        if (iteration_num > (net.max_batches * 1 / 2)) l.deform = 0;
        if (l.burnin_update && (l.burnin_update*net.burn_in > iteration_num)) continue;
        if (l.train_only_bn) continue;

        if(l.update_gpu && l.dont_update < iteration_num){
            l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
        }
    }
}

void forward_backward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    if (net.adversarial) {
        state.delta = cuda_make_array(NULL, x_size);
    }
    state.truth = *net.truth_gpu;
    state.train = 1;
#if defined(CUDNN_HALF) && defined(CUDNN)
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (net.cudnn_half){
            if (l.type == CONVOLUTIONAL && l.weights_gpu && l.weights_gpu16) {
                assert((l.nweights) > 0);
                cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
            }
            else if (l.type == CRNN && l.input_layer->weights_gpu && l.input_layer->weights_gpu16) {
                assert((l.input_layer->c*l.input_layer->n*l.input_layer->size*l.input_layer->size) > 0);
                cuda_convert_f32_to_f16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
                cuda_convert_f32_to_f16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
                cuda_convert_f32_to_f16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
            }
            else if (l.type == CONV_LSTM && l.wf->weights_gpu && l.wf->weights_gpu16) {
                assert((l.wf->c * l.wf->n * l.wf->size * l.wf->size) > 0);
                if (l.peephole) {
                    cuda_convert_f32_to_f16(l.vf->weights_gpu, l.vf->nweights, l.vf->weights_gpu16);
                    cuda_convert_f32_to_f16(l.vi->weights_gpu, l.vi->nweights, l.vi->weights_gpu16);
                    cuda_convert_f32_to_f16(l.vo->weights_gpu, l.vo->nweights, l.vo->weights_gpu16);
                }
                cuda_convert_f32_to_f16(l.wf->weights_gpu, l.wf->nweights, l.wf->weights_gpu16);
                if (!l.bottleneck) {
                    cuda_convert_f32_to_f16(l.wi->weights_gpu, l.wi->nweights, l.wi->weights_gpu16);
                    cuda_convert_f32_to_f16(l.wg->weights_gpu, l.wg->nweights, l.wg->weights_gpu16);
                    cuda_convert_f32_to_f16(l.wo->weights_gpu, l.wo->nweights, l.wo->weights_gpu16);
                }
                cuda_convert_f32_to_f16(l.uf->weights_gpu, l.uf->nweights, l.uf->weights_gpu16);
                cuda_convert_f32_to_f16(l.ui->weights_gpu, l.ui->nweights, l.ui->weights_gpu16);
                cuda_convert_f32_to_f16(l.ug->weights_gpu, l.ug->nweights, l.ug->weights_gpu16);
                cuda_convert_f32_to_f16(l.uo->weights_gpu, l.uo->nweights, l.uo->weights_gpu16);
            }
        }
    }
#endif
    forward_network_gpu(net, state);
    //cudaStreamSynchronize(get_cuda_stream());
    backward_network_gpu(net, state);

    if (net.adversarial) {
        cuda_free(state.delta);
        cuda_pull_array(*net.input_gpu, x, x_size);
    }
    if(*(state.net.total_bbox) > 0)
        fprintf(stderr, " total_bbox = %d, rewritten_bbox = %f %% \n", *(state.net.total_bbox), 100 * (float)*(state.net.rewritten_bbox) / *(state.net.total_bbox));
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    *net.seen += net.batch;
    if (net.adversarial_lr && rand_int(0, 1) == 1 && get_current_iteration(net) > net.burn_in) {
        net.adversarial = 1;
        float lr_old = net.learning_rate;
        float scale = (get_current_iteration(net) / ((float)net.max_batches));
        //scale = sin(scale * M_PI);
        net.learning_rate = net.adversarial_lr * scale;
        layer l = net.layers[net.n - 1];
        int y_size = get_network_output_size(net)*net.batch;
        if (net.layers[net.n - 1].truths) y_size = net.layers[net.n - 1].truths*net.batch;
        float *truth_cpu = (float *)xcalloc(y_size, sizeof(float));

        const int img_size = net.w*net.h*net.c;
        float *old_input = (float *)xcalloc(img_size*net.batch, sizeof(float));
        memcpy(old_input, x, img_size*net.batch * sizeof(float));

        printf("\n adversarial training, adversarial_lr = %f \n", net.adversarial_lr * scale);

        forward_backward_network_gpu(net, x, truth_cpu);

        int b;
        for (b = 0; b < net.batch; ++b) {
            if (b % 2 == 1 && net.contrastive) {
                //printf(" b = %d old img, ", b);
                memcpy(x + img_size*b, old_input + img_size*b, img_size * sizeof(float));
            }
        }

        image im;
        im.w = net.w;
        im.h = net.h;
        im.c = net.c;
        im.data = x;
        show_image(im, "adversarial data augmentation");
        resize_window_cv("adversarial data augmentation", 500, 500);
        wait_key_cv(1);

        free(old_input);
        free(truth_cpu);
        net.learning_rate = lr_old;
        net.adversarial = 0;
    }
    forward_backward_network_gpu(net, x, y);
    float error = get_network_cost(net);
    //if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);
    const int sequence = get_sequence_value(net);
    //if (((*net.seen) / net.batch) % (net.subdivisions*sequence) == 0) update_network_gpu(net);

    return error;
}

typedef struct {
    network net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net.gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed", DARKNET_LOC);
    return thread;
}

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void update_layer(layer l, network net)
{
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    l.t = get_current_batch(net);
    if(l.update_gpu){
        l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
    }
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weights, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.biases, l.n);
        cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if(base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
        if (l.scale_updates) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
    }
}

void distribute_updates(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
        if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}

void sync_layer(network *nets, int n, int j)
{
    //printf("Syncing layer %d\n", j);
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    cuda_set_device(net.gpu_index);
    pull_weights(base);
    for (i = 1; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
    //printf("Done syncing layer %d\n", j);
}

typedef struct{
    network *nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed", DARKNET_LOC);
    return thread;
}

void sync_nets(network *nets, int n, int interval)
{
    int j;
    int layers = nets[0].n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
    for (j = 0; j < n; ++j){
        *nets[j].seen = *nets[0].seen;
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network *nets, int n, data d, int interval)
{
    int i;
#ifdef _DEBUG
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
#endif
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    *nets[0].cur_iteration += (n - 1);
    *nets[0].seen = nets[0].batch * nets[0].subdivisions * get_current_iteration(nets[0]); // remove this line, when you will save to weights-file both: seen & cur_iteration
    if (get_current_iteration(nets[0]) % interval == 0)
    {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    if(l.type != REGION && l.type != YOLO && (*net.cuda_graph_ready) == 0) cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    if (net.gpu_index != cuda_get_device())
        cuda_set_device(net.gpu_index);
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    //state.input = cuda_make_array(input, size);   // memory will be allocated in the parse_network_cfg_custom()
    state.input = net.input_state_gpu;
    memcpy(net.input_pinned_cpu, input, size * sizeof(float));
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    //cudaGraphExec_t instance = (cudaGraphExec_t)net.cuda_graph_exec;
    static cudaGraphExec_t instance;

    if ((*net.cuda_graph_ready) == 0) {
        static cudaGraph_t graph;
        if (net.use_cuda_graph == 1) {
            int i;
            for (i = 0; i < 16; ++i) switch_stream(i);

            cudaStream_t stream0 = switch_stream(0);
            CHECK_CUDA(cudaDeviceSynchronize());
            printf("Try to capture graph... \n");
            //cudaGraph_t graph = (cudaGraph_t)net.cuda_graph;
            CHECK_CUDA(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));
        }

        cuda_push_array(state.input, net.input_pinned_cpu, size);
        forward_network_gpu(net, state);

        if (net.use_cuda_graph == 1) {
            cudaStream_t stream0 = switch_stream(0);
            CHECK_CUDA(cudaStreamEndCapture(stream0, &graph));
            CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
            (*net.cuda_graph_ready) = 1;
            printf(" graph is captured... \n");
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
    }
    else {
        cudaStream_t stream0 = switch_stream(0);
        //printf(" cudaGraphLaunch \n");
        CHECK_CUDA( cudaGraphLaunch(instance, stream0) );
        CHECK_CUDA( cudaStreamSynchronize(stream0) );
        //printf(" ~cudaGraphLaunch \n");
    }

    float *out = get_network_output_gpu(net);
    reset_wait_stream_events();
    //cuda_free(state.input);   // will be freed in the free_network()
    return out;
}
