#include "activations.h"

#include <math.h>

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
    return (x>=0);
}

double sigmoid_activation(double x)
{
    return 1./(1.+exp(-x));
}

double sigmoid_gradient(double x)
{
    return x*(1.-x);
}

