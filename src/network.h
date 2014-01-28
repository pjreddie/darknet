// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "data.h"

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX
} LAYER_TYPE;

typedef struct {
    int n;
    void **layers;
    LAYER_TYPE *types;
    int outputs;
    double *output;
} network;

network make_network(int n);
void forward_network(network net, double *input);
double backward_network(network net, double *input, double *truth);
void update_network(network net, double step, double momentum, double decay);
double train_network_sgd(network net, data d, int n, double step, double momentum,double decay);
double train_network_batch(network net, data d, int n, double step, double momentum,double decay);
void train_network(network net, data d, double step, double momentum, double decay);
matrix network_predict_data(network net, data test);
double network_accuracy(network net, data d);
double *get_network_output(network net);
double *get_network_output_layer(network net, int i);
double *get_network_delta_layer(network net, int i);
double *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);

#endif

