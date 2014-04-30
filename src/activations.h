#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef enum{
    SIGMOID, RELU, LINEAR, RAMP, TANH
}ACTIVATION;

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void activate_array(float *x, const int n, const ACTIVATION a);

#endif

