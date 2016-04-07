#ifndef RBM_LAYER_H
#define RBM_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer rbm_layer;

rbm_layer make_rbm_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize);

void preforward_rbm_layer(rbm_layer layer, network_state state);
void prebackward_rbm_layer(rbm_layer layer, network_state state);
void preupdate_rbm_layer(rbm_layer l, int batch, float learning_rate, float momentum, float decay);
void mirror_rbm_layer(rbm_layer l, rbm_layer r);

void forward_rbm_layer(rbm_layer l, network_state state);

#ifdef GPU
void preforward_rbm_layer_gpu(rbm_layer layer, network_state state);
void prebackward_rbm_layer_gpu(rbm_layer layer, network_state state);
void preupdate_rbm_layer_gpu(rbm_layer layer, int batch, float learning_rate, float momentum, float decay);
void mirror_rbm_layer_gpu(rbm_layer l, rbm_layer r);

void forward_rbm_layer_gpu(rbm_layer l, network_state state);

void push_rbm_layer(rbm_layer layer);
void pull_rbm_layer(rbm_layer layer);
#endif

#endif

