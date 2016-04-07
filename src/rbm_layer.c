#include "rbm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"
#include "connected_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

rbm_layer make_rbm_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize)
{
    int i;
    rbm_layer l = {0};
    l.type = RBM;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.weights = calloc(outputs*inputs, sizeof(float));

    l.bias_updates = calloc(outputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));

    //for pretrain
    l.vbiases = calloc(inputs, sizeof(float));
    l.vbias_updates = calloc(inputs, sizeof(float));
    l.vinput = calloc(batch*inputs, sizeof(float));
    l.houtput = calloc(batch*outputs, sizeof(float));
 
    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = 2*scale*rand_uniform(-1, 1) - scale;
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = scale;
    }

    for(i=0; i < inputs; ++i){
        l.vbiases[i] = scale;
    }

    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);

    //for pretrain
    l.vbiases_gpu = cuda_make_array(l.vbiases, inputs);
    l.vbias_updates_gpu = cuda_make_array(l.vbias_updates, inputs);
    l.vinput_gpu = cuda_make_array(l.vinput, batch*inputs);
    l.houtput_gpu = cuda_make_array(l.houtput, batch*outputs);

    if(batch_normalize){
        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
    }

#endif

    l.activation = activation;
    fprintf(stderr, "RBM Layer: %d inputs, %d outputs\n", l.inputs, l.outputs);
    return l;
}

void activate_hidden1(rbm_layer l, network_state state) {
    //firstly copy biases to the output
    int i;
    for(i = 0; i < l.batch; ++i){
        copy_cpu(l.outputs, l.biases, 1, l.output + i*l.outputs, 1); // in blas
    }
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = state.input; 
    float *c = l.output;
    float *b = l.weights;
    //secondly add input*weight to the output
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n); // in gemm
    //thirdly activate the output
    activate_array(l.output, l.outputs*l.batch, l.activation); // in activation
}
void activate_hidden2(rbm_layer l) {
    //firstly copy biases to the houtput
    int i;
    for(i = 0; i < l.batch; ++i){
        copy_cpu(l.outputs, l.biases, 1, l.houtput + i*l.outputs, 1); // in blas
    }
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = l.vinput; 
    float *c = l.houtput;
    float *b = l.weights;
    //secondly add input*weight to the houtput
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n); // in gemm
    //thirdly activate the houtput
    activate_array(l.houtput, l.outputs*l.batch, l.activation); // in activation
}
void activate_visible(rbm_layer l) {
    //firstly copy vbiases to the voutput
    int i;
    for(i = 0; i < l.batch; ++i) {
        copy_cpu(l.inputs, l.vbiases, 1, l.vinput + i*l.inputs, 1);
    }
    //secondly add output*weight to the voutput
    int m = l.batch;
    int k = l.outputs;
    int n = l.inputs;
    float* a = l.output;

    //for(i=0; i<l.batch*l.outputs; ++i){
    //    a[i] = (rand_uniform<a[i])? 1.0 : 0;
    //}

    float* b = l.weights;
    float* c = l.vinput;
    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n); // in gemm
    //thirdly activate the output
    activate_array(l.vinput, l.inputs*l.batch, l.activation); // in activation
}
void preforward_rbm_layer(rbm_layer l, network_state state) {
    activate_hidden1(l, state);
    activate_visible(l);
    activate_hidden2(l);
}

void rbm_axpy_cpu(int N, float ALPHA, float BETAX, float BETAY, float *X, int INCX, float *Y, int INCY, float *Z, int INCZ) {
    int i;
    for(i = 0; i < N; ++i) Z[i*INCZ] += ALPHA*(BETAX*X[i*INCX]+BETAY*Y[i*INCY]);
}
void rbm_gemm_cpu(int outputs, int inputs, float* h1, float* h2, float* v1, float* v2, float* weights) {
    int i, j;
    for(i=0; i<outputs; i++) {
        for(j=0; j<inputs; j++) {
            weights[i*inputs+j] += h1[i]*v1[j] - h2[i]*v2[j];
        }
    }
}
void prebackward_rbm_layer(rbm_layer l, network_state state) {
    //compute updates
    int i;
    for(i=0; i<l.batch; i++) {
        rbm_axpy_cpu(l.outputs, 1, 1, -1, l.output + i*l.outputs, 1, l.houtput + i*l.outputs, 1, l.bias_updates, 1); 
        rbm_axpy_cpu(l.inputs, 1, 1, -1, state.input + i*l.inputs, 1, l.vinput + i*l.inputs, 1, l.vbias_updates, 1);
        rbm_gemm_cpu(l.outputs, l.inputs, l.output + i*l.outputs, l.houtput + i*l.outputs, 
					 state.input + i*l.inputs, l.vinput + i*l.inputs, l.weight_updates);
    }
}

void preupdate_rbm_layer(rbm_layer l, int batch, float learning_rate, float momentum, float decay) {
    // all in blas
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    axpy_cpu(l.inputs, learning_rate/batch, l.vbias_updates, 1, l.vbiases, 1);
    scal_cpu(l.inputs, momentum, l.vbias_updates, 1);

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void mirror_rbm_layer(rbm_layer l, rbm_layer r) {
    assert(l.inputs == r.outputs && l.outputs == r.inputs);
    int i, j;
    for(i=0; i<l.outputs; i++) {
        for(j=0; j<l.inputs; j++) {
            r.weights[j*r.inputs+i] = l.weights[i*l.inputs+j];
        }
    }
    for(i=0; i<r.outputs; i++) {
        r.biases[i] = l.vbiases[i];
    }
    for(i=0; i<r.inputs; i++) {
        r.vbiases[i] = l.biases[i];
    }
}

void forward_rbm_layer(rbm_layer l, network_state state)
{
    l.type = CONNECTED;
    forward_connected_layer(l, state);
    l.type = RBM;
}



