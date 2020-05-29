#ifndef CONV_LSTM_LAYER_H
#define CONV_LSTM_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET

#ifdef __cplusplus
extern "C" {
#endif
layer make_conv_lstm_layer(int batch, int h, int w, int c, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int peephole, int xnor, int bottleneck, int train);
void resize_conv_lstm_layer(layer *l, int w, int h);
void free_state_conv_lstm(layer l);
void randomize_state_conv_lstm(layer l);
void remember_state_conv_lstm(layer l);
void restore_state_conv_lstm(layer l);

void forward_conv_lstm_layer(layer l, network_state state);
void backward_conv_lstm_layer(layer l, network_state state);
void update_conv_lstm_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_conv_lstm_layer_gpu(layer l, network_state state);
void backward_conv_lstm_layer_gpu(layer l, network_state state);
void update_conv_lstm_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
#endif

#ifdef __cplusplus
}
#endif

#endif  // CONV_LSTM_LAYER_H
