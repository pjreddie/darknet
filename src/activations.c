#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case GELU:
            return "gelu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "swish") == 0) return SWISH;
    if (strcmp(s, "mish") == 0) return MISH;
    if (strcmp(s, "hard_mish") == 0) return HARD_MISH;
    if (strcmp(s, "normalize_channels") == 0) return NORM_CHAN;
    if (strcmp(s, "normalize_channels_softmax") == 0) return NORM_CHAN_SOFTMAX;
    if (strcmp(s, "normalize_channels_softmax_maxval") == 0) return NORM_CHAN_SOFTMAX_MAXVAL;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "relu6") == 0) return RELU6;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu") == 0) return SELU;
    if (strcmp(s, "gelu") == 0) return GELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "revleaky") == 0) return REVLEAKY;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case GELU:
            return gelu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case REVLEAKY:
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR) {}
    else if (a == LEAKY) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = leaky_activate(x[i]);
        }
    }
    else if (a == LOGISTIC) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = logistic_activate(x[i]);
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}

void activate_array_swish(float *x, const int n, float * output_sigmoid, float * output)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float x_val = x[i];
        float sigmoid = logistic_activate(x_val);
        output_sigmoid[i] = sigmoid;
        output[i] = x_val * sigmoid;
    }
}

// https://github.com/digantamisra98/Mish
void activate_array_mish(float *x, const int n, float * activation_input, float * output)
{
    const float MISH_THRESHOLD = 20;
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float x_val = x[i];
        activation_input[i] = x_val;    // store value before activation
        output[i] = x_val * tanh_activate( softplus_activate(x_val, MISH_THRESHOLD) );
    }
}

static float hard_mish_yashas(float x)
{
    if (x > 0)
        return x;
    if (x > -2)
        return x * x / 2 + x;
    return 0;
}

void activate_array_hard_mish(float *x, const int n, float * activation_input, float * output)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float x_val = x[i];
        activation_input[i] = x_val;    // store value before activation
        output[i] = hard_mish_yashas(x_val);
    }
}

void activate_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *output)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        const float eps = 0.0001;
        if (i < size) {
            float sum = eps;
            int k;
            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                if (val > 0) sum += val;
            }
            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                if (val > 0) val = val / sum;
                else val = 0;
                output[wh_i + k * wh_step + b*wh_step*channels] = val;
            }
        }
    }
}

void activate_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *output, int use_max_val)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        const float eps = 0.0001;
        if (i < size) {
            float sum = eps;
            float max_val = -FLT_MAX;
            int k;
            if (use_max_val) {
                for (k = 0; k < channels; ++k) {
                    float val = x[wh_i + k * wh_step + b*wh_step*channels];
                    if (val > max_val || k == 0) max_val = val;
                }
            }
            else
                max_val = 0;

            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                sum += expf(val - max_val);
            }
            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                val = expf(val - max_val) / sum;
                output[wh_i + k * wh_step + b*wh_step*channels] = val;
            }
        }
    }
}

void gradient_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float out = x[index];
                float d = delta[index];
                grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float d = delta[index];
                d = d * grad;
                delta[index] = d;
            }
        }
    }
}

void gradient_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float out = x[index];
                float d = delta[index];
                grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                if (x[index] > 0) {
                    float d = delta[index];
                    d = d * grad;
                    delta[index] = d;
                }
            }
        }
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case RELU6:
            return relu6_gradient(x);
        case NORM_CHAN:
            //return relu_gradient(x);
        case NORM_CHAN_SOFTMAX_MAXVAL:
            //...
        case NORM_CHAN_SOFTMAX:
            error("Error: should be used custom NORM_CHAN or NORM_CHAN_SOFTMAX-function for gradient", DARKNET_LOC);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case GELU:
            return gelu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case REVLEAKY:
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}

// https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cpp#L54-L56
void gradient_array_swish(const float *x, const int n, const float * sigmoid, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float swish = x[i];
        delta[i] *= swish + sigmoid[i]*(1 - swish);
    }
}

// https://github.com/digantamisra98/Mish
void gradient_array_mish(const int n, const float * activation_input, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        const float MISH_THRESHOLD = 20.0f;

        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        float inp = activation_input[i];
        const float sp = softplus_activate(inp, MISH_THRESHOLD);
        const float grad_sp = 1 - exp(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        delta[i] *= grad;


        //float x = activation_input[i];
        //float d = 2 * expf(x) + expf(2 * x) + 2;
        //float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x)*(4 * x + 6);
        //float derivative = expf(x) * w / (d * d);
        //delta[i] *= derivative;
    }
}

static float hard_mish_yashas_grad(float x)
{
    if (x > 0)
        return 1;
    if (x > -2)
        return x + 1;
    return 0;
}

void gradient_array_hard_mish(const int n, const float * activation_input, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float inp = activation_input[i];
        delta[i] *= hard_mish_yashas_grad(inp);
    }
}
