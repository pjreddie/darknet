// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;
    int batch;
    int *seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;

    int gpu_index;
    tree *hierarchy;



    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;

    #ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
    #endif

} network;


#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net);
float *network_predict_gpu(network net, float *input);
void pull_network_output(network net);
void forward_network_gpu(network net);
void backward_network_gpu(network net);
void update_network_gpu(network net);
void harmless_update_network_gpu(network net);
#endif

float get_current_rate(network net);
int get_current_batch(network net);
void free_network(network net);
void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net);
void backward_network(network net);
void update_network(network net);

float train_network(network net, data d);
float train_network_sgd(network net, data d, int n);
float train_network_datum(network net);

matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
layer get_network_output_layer(network net);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
network load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network net);
void calc_network_cost(network net);

#endif

