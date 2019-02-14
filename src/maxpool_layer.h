#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer maxpool_layer;

#ifdef __cplusplus
extern "C" {
#endif
image get_maxpool_image(maxpool_layer l);
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network_state state);
void backward_maxpool_layer(const maxpool_layer l, network_state state);

#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer l, network_state state);
void backward_maxpool_layer_gpu(maxpool_layer l, network_state state);
void cudnn_maxpool_setup(maxpool_layer *l);
#endif // GPU

#ifdef __cplusplus
}
#endif

#endif
