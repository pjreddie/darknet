// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "opencl.h"
#include "image.h"
#include "data.h"

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    NORMALIZATION,
    DROPOUT,
    FREEWEIGHT,
    CROP,
    COST
} LAYER_TYPE;

typedef struct {
    int n;
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    void **layers;
    LAYER_TYPE *types;
    int outputs;
    float *output;

    #ifdef GPU
    cl_mem *input_cl;
    cl_mem *truth_cl;
    #endif
} network;

#ifdef GPU
void forward_network_gpu(network net, cl_mem input, cl_mem truth, int train);
void backward_network_gpu(network net, cl_mem input);
void update_network_gpu(network net);
cl_mem get_network_output_cl_layer(network net, int i);
cl_mem get_network_delta_cl_layer(network net, int i);
float train_network_sgd_gpu(network net, data d, int n);
float train_network_data_gpu(network net, data d, int n);
#endif

network make_network(int n, int batch);
void forward_network(network net, float *input, float *truth, int train);
void backward_network(network net, float *input);
void update_network(network net);
float train_network_sgd(network net, data d, int n);
float train_network_batch(network net, data d, int n);
void train_network(network net, data d);
matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network net, int h, int w, int c);
int get_network_input_size(network net);
float get_network_cost(network net);

#endif

