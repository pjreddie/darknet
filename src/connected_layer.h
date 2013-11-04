#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

typedef struct{
    int inputs;
    int outputs;
    double *weights;
    double *biases;
    double *weight_updates;
    double *bias_updates;
    double *output;
} connected_layer;

connected_layer make_connected_layer(int inputs, int outputs);
void run_connected_layer(double *input, connected_layer layer);
void backpropagate_connected_layer(double *input, connected_layer layer);
void calculate_updates_connected_layer(double *input, connected_layer layer);
void update_connected_layer(connected_layer layer, double step);

#endif

