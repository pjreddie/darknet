#ifndef SAM_CHANNELS_LAYER_H
#define SAM_CHANNELS_LAYER_H

#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_sam_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_sam_layer(const layer l, network_state state);
void backward_sam_layer(const layer l, network_state state);
void resize_sam_layer(layer *l, int w, int h);

#ifdef GPU
void forward_sam_layer_gpu(const layer l, network_state state);
void backward_sam_layer_gpu(const layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif  // SAM_CHANNELS_LAYER_H
