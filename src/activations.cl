typedef enum{
    SIGMOID, RELU, LINEAR, RAMP, TANH
}ACTIVATION;

float activate(float x, ACTIVATION a, float dropout)
{
    //if((float)rand()/RAND_MAX < dropout) return 0;
    switch(a){
        case LINEAR:
            return linear_activate(x)/(1-dropout);
        case SIGMOID:
            return sigmoid_activate(x)/(1-dropout);
        case RELU:
            return relu_activate(x)/(1-dropout);
        case RAMP:
            return ramp_activate(x)/(1-dropout);
        case TANH:
            return tanh_activate(x)/(1-dropout);
    }
    return 0;
}

__kernel void activate_array(__global float *x,
    const int n, const ACTIVATION a, const float dropout)
{
    int i = get_global_id(0);
    x[i] = activate(x[i], a, dropout);
}
