#include "cuda.h"
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
#ifdef GPU
void activate_array_ongpu(float *x, int n, ACTIVATION a);
void gradient_array_ongpu(float *x, int n, ACTIVATION a, float *delta);
#endif

static inline float linear_activate(float x){return x;}
static inline float sigmoid_activate(float x){return 1./(1. + exp(-x));}
static inline float relu_activate(float x){return x*(x>0);}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

static inline float linear_gradient(float x){return 1;}
static inline float sigmoid_gradient(float x){return (1-x)*x;}
static inline float relu_gradient(float x){return (x>0);}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float tanh_gradient(float x){return 1-x*x;}

#endif

