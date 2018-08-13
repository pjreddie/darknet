#ifndef CONVERTER_LAYER_H
#define CONVERTER_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer converter_layer;

converter_layer make_converter_layer(int batch, int w, int h, int c,
                                        converter_params params);
void forward_converter_layer(const converter_layer l, network net);
void backward_converter_layer(const converter_layer l, network net);
PRECISION get_precision(const char *precision);

#endif
