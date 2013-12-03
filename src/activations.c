#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "sigmoid")==0) return SIGMOID;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "identity")==0) return IDENTITY;
    if (strcmp(s, "ramp")==0) return RAMP;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

double activate(double x, ACTIVATION a){
    switch(a){
        case IDENTITY:
            return x;
        case SIGMOID:
            return 1./(1.+exp(-x));
        case RELU:
            return x*(x>0);
        case RAMP:
            return x*(x>0) + .1*x;
    }
    return 0;
}
double gradient(double x, ACTIVATION a){
    switch(a){
        case IDENTITY:
            return 1;
        case SIGMOID:
            return (1.-x)*x;
        case RELU:
            return (x>0);
        case RAMP:
            return (x>0) + .1;
    }
    return 0;
}

double identity_activation(double x)
{
    return x;
}
double identity_gradient(double x)
{
    return 1;
}

double relu_activation(double x)
{
    return x*(x>0);
}
double relu_gradient(double x)
{
    return (x>0);
}

double sigmoid_activation(double x)
{
    return 1./(1.+exp(-x));
}

double sigmoid_gradient(double x)
{
    return x*(1.-x);
}

