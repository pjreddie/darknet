#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
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
}

void forward_network_gpu(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(net);
    calc_network_cost(net);
}

void backward_network_gpu(network net)
{
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(net);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

void harmless_update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

float train_network_datum_gpu(network net)
{
    *net.seen += net.batch;

    int x_size = net.inputs*net.batch;
    int y_size = net.truths*net.batch;
    cuda_push_array(net.input_gpu, net.input, x_size);
    cuda_push_array(net.truth_gpu, net.truth, y_size);

    net.train = 1;
    forward_network_gpu(net);
    backward_network_gpu(net);

    float error = *net.cost;
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);

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
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.n*l.size*l.size*l.c, s, l.weights, 1);
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
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.n*l.size*l.size*l.c);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
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
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
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
        l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
    }
}
void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weight_updates, 1, base.weight_updates, 1);
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
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.n*l.size*l.size*l.c);
        if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}
*/

/*
void sync_layer(network *nets, int n, int j)
{
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
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
}
*/

void sync_layer(network *nets, int n, int j)
{
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
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
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
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
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
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
    if (get_current_batch(nets[0]) % interval == 0) {
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

void pull_network_output(network net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

float *network_predict_gpu(network net, float *input)
{
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, input, net.inputs*net.batch);
    net.truth = 0;
    net.train = 0;
    forward_network_gpu(net);
    return net.output;
}

