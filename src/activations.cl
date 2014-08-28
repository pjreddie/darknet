typedef enum{
    SIGMOID, RELU, LINEAR, RAMP, TANH
}ACTIVATION;

float linear_activate(float x){return x;}
float sigmoid_activate(float x){return 1./(1. + exp(-x));}
float relu_activate(float x){return x*(x>0);}
float ramp_activate(float x){return x*(x>0)+.1*x;}
float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

float linear_gradient(float x){return 1;}
float sigmoid_gradient(float x){return (1-x)*x;}
float relu_gradient(float x){return (x>0);}
float ramp_gradient(float x){return (x>0)+.1;}
float tanh_gradient(float x){return 1-x*x;}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case SIGMOID:
            return sigmoid_activate(x);
        case RELU:
            return relu_activate(x);
        case RAMP:
            return ramp_activate(x);
        case TANH:
            return tanh_activate(x);
    }
    return 0;
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case SIGMOID:
            return sigmoid_gradient(x);
        case RELU:
            return relu_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case TANH:
            return tanh_gradient(x);
    }
    return 0;
}

__kernel void activate_array(__global float *x, int n, ACTIVATION a)
{
    int i = get_global_id(0);
    x[i] = activate(x[i], a);
}

__kernel void gradient_array(__global float *x, int n, ACTIVATION a, __global float *delta)
{
    int i = get_global_id(0);
    delta[i] *= gradient(x[i], a);
}

