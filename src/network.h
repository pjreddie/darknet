// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "data.h"

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    NORMALIZATION
} LAYER_TYPE;

typedef struct {
    int n;
    int batch;
    void **layers;
    LAYER_TYPE *types;
    int outputs;
    float *output;
} network;

network make_network(int n, int batch);
void forward_network(network net, float *input);
float backward_network(network net, float *input, float *truth);
void update_network(network net, float step, float momentum, float decay);
float train_network_sgd(network net, data d, int n, float step, float momentum,float decay);
float train_network_batch(network net, data d, int n, float step, float momentum,float decay);
void train_network(network net, data d, float step, float momentum, float decay);
matrix network_predict_data(network net, data test);
float network_accuracy(network net, data d);
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
void save_network(network net, char *filename);
int resize_network(network net, int h, int w, int c);

#endif

