#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "sigmoid")==0) return SIGMOID;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "tanh")==0) return TANH;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a){
    switch(a){
        case LINEAR:
            return x;
        case SIGMOID:
            return 1./(1.+exp(-x));
        case RELU:
            return x*(x>0);
        case RAMP:
            return x*(x>0) + .1*x;
        case TANH:
            return (exp(2*x)-1)/(exp(2*x)+1);
    }
    return 0;
}
float gradient(float x, ACTIVATION a){
    switch(a){
        case LINEAR:
            return 1;
        case SIGMOID:
            return (1.-x)*x;
        case RELU:
            return (x>0);
        case RAMP:
            return (x>0) + .1;
        case TANH:
            return 1-x*x;
    }
    return 0;
}

