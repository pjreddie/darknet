#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef enum{
    SIGMOID, RELU, IDENTITY, RAMP
}ACTIVATION;

ACTIVATION get_activation(char *s);

double activate(double x, ACTIVATION a);
double gradient(double x, ACTIVATION a);

#endif

