#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

void save_network(network net, const char *filename);
void save_weights_double(network net, const char *filename);

#endif
