#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

void save_network(dn_network net, const char *filename);
void save_weights_double(dn_network net, const char *filename);

#endif
