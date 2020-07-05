#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;
typedef layer contrastive_layer;

#ifdef __cplusplus
extern "C" {
#endif
void softmax_array(float *input, int n, float temp, float *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network_state state);
void backward_softmax_layer(const softmax_layer l, network_state state);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network_state state);
void backward_softmax_layer_gpu(const softmax_layer l, network_state state);
#endif

//-----------------------

contrastive_layer make_contrastive_layer(int batch, int w, int h, int n, int classes, int inputs, layer *yolo_layer);
void forward_contrastive_layer(contrastive_layer l, network_state state);
void backward_contrastive_layer(contrastive_layer l, network_state net);

#ifdef GPU
void pull_contrastive_layer_output(const contrastive_layer l);
void push_contrastive_layer_output(const contrastive_layer l);
void forward_contrastive_layer_gpu(contrastive_layer l, network_state state);
void backward_contrastive_layer_gpu(contrastive_layer layer, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
