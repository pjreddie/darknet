#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2, int assisted_excitation, int train);
void forward_shortcut_layer(const layer l, network_state state);
void backward_shortcut_layer(const layer l, network_state state);
void resize_shortcut_layer(layer *l, int w, int h);

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network_state state);
void backward_shortcut_layer_gpu(const layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
