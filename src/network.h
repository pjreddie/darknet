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
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
#endif

void compare_networks(network n1, network n2, data d);

network make_network(int n, int batch);
void forward_network(network net, float *input, float *truth, int train);
void backward_network(network net, float *input);
void update_network(network net);

float train_network(network net, data d);
float train_network_batch(network net, data d, int n);
float train_network_sgd(network net, data d, int n);

matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d);
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
void set_batch_network(network *net, int b);
void set_learning_network(network *net, float rate, float momentum, float decay);
int get_network_input_size(network net);
float get_network_cost(network net);

#endif

