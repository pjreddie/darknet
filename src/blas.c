#include "blas.h"
#include "utils.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int out_w, int out_h, int out_c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int in_c = out_c/(stride*stride);

    //printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
    //printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w*stride, out_h*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < out_c; ++k){
            for(j = 0; j < out_h; ++j){
                for(i = 0; i < out_w; ++i){
                    int in_index  = i + out_w*(j + out_h*(k + out_c*b));
                    int c2 = k % in_c;
                    int offset = k / in_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + out_w*stride*(h2 + out_h*stride*(c2 + in_c*b));
                    if(forward) out[out_index] = x[in_index];    // used by default for forward (i.e. forward = 0)
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float* swap = (float*)xcalloc(size * layers * batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

static float relu(float src) {
    if (src > 0) return src;
    return 0;
}

void shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers, float **layers_output, float *out, float *in, float *weights, int nweights, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {

        int src_id = id;
        const int src_i = src_id % src_outputs;
        src_id /= src_outputs;
        int src_b = src_id;

        float sum = 1, max_val = -FLT_MAX;
        int i;
        if (weights && weights_normalization) {
            if (weights_normalization == SOFTMAX_NORMALIZATION) {
                for (i = 0; i < (n + 1); ++i) {
                    const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (max_val < w) max_val = w;
                }
            }
            const float eps = 0.0001;
            sum = eps;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) sum += relu(w);
                else if (weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
            }
        }

        if (weights) {
            float w = weights[src_i / step];
            if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
            else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

            out[id] = in[id] * w; // [0 or c or (c, h ,w)]
        }
        else out[id] = in[id];

        // layers
        for (i = 0; i < n; ++i) {
            int add_outputs = outputs_of_layers[i];
            if (src_i < add_outputs) {
                int add_index = add_outputs*src_b + src_i;
                int out_index = id;

                float *add = layers_output[i];

                if (weights) {
                    const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
                    else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                    out[out_index] += add[add_index] * w; // [0 or c or (c, h ,w)]
                }
                else out[out_index] += add[add_index];
            }
        }
    }
}

void backward_shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers,
    float **layers_delta, float *delta_out, float *delta_in, float *weights, float *weight_updates, int nweights, float *in, float **layers_output, WEIGHTS_NORMALIZATION_T weights_normalization)
{
    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {
        int src_id = id;
        int src_i = src_id % src_outputs;
        src_id /= src_outputs;
        int src_b = src_id;

        float grad = 1, sum = 1, max_val = -FLT_MAX;;
        int i;
        if (weights && weights_normalization) {
            if (weights_normalization == SOFTMAX_NORMALIZATION) {
                for (i = 0; i < (n + 1); ++i) {
                    const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (max_val < w) max_val = w;
                }
            }
            const float eps = 0.0001;
            sum = eps;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) sum += relu(w);
                else if (weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
            }

            /*
            grad = 0;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float delta_w = delta_in[id] * in[id];
                const float w = weights[weights_index];
                if (weights_normalization == RELU_NORMALIZATION) grad += delta_w * relu(w) / sum;
                else if (weights_normalization == SOFTMAX_NORMALIZATION) grad += delta_w * expf(w - max_val) / sum;
            }
            */
        }

        if (weights) {
            float w = weights[src_i / step];
            if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
            else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

            delta_out[id] += delta_in[id] * w; // [0 or c or (c, h ,w)]
            weight_updates[src_i / step] += delta_in[id] * in[id] * grad;
        }
        else delta_out[id] += delta_in[id];

        // layers
        for (i = 0; i < n; ++i) {
            int add_outputs = outputs_of_layers[i];
            if (src_i < add_outputs) {
                int add_index = add_outputs*src_b + src_i;
                int out_index = id;

                float *layer_delta = layers_delta[i];
                if (weights) {
                    float *add = layers_output[i];

                    const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (weights_normalization == RELU_NORMALIZATION) w = relu(w) / sum;
                    else if (weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                    layer_delta[add_index] += delta_in[id] * w; // [0 or c or (c, h ,w)]
                    weight_updates[weights_index] += delta_in[id] * add[add_index] * grad;
                }
                else layer_delta[add_index] += delta_in[id];
            }
        }
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .000001f));
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    if (INCX == 1 && ALPHA == 0) {
        memset(X, 0, N * sizeof(float));
    }
    else {
        for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
    }
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, float *output, int stride)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, output + b*batch_offset + g*group_offset, stride);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h*stride; ++j) {
                for (i = 0; i < w*stride; ++i) {
                    int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if (forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}


void constrain_cpu(int size, float ALPHA, float *X)
{
    int i;
    for (i = 0; i < size; ++i) {
        X[i] = fminf(ALPHA, fmaxf(-ALPHA, X[i]));
    }
}

void fix_nan_and_inf_cpu(float *input, size_t size)
{
    int i;
    for (i = 0; i < size; ++i) {
        float val = input[i];
        if (isnan(val) || isinf(val))
            input[i] = 1.0f / i;  // pseudo random value
    }
}
