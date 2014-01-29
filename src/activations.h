#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef enum{
    SIGMOID, RELU, LINEAR, RAMP, TANH
}ACTIVATION;

ACTIVATION get_activation(char *s);

float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);

#endif

