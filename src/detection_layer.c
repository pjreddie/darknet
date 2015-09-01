#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int get_detection_layer_locations(detection_layer l)
{
    return l.inputs / (l.classes+l.coords+l.joint+(l.background || l.objectness));
}

int get_detection_layer_output_size(detection_layer l)
{
    return get_detection_layer_locations(l)*((l.background || l.objectness) + l.classes + l.coords);
}

detection_layer make_detection_layer(int batch, int inputs, int classes, int coords, int joint, int rescore, int background, int objectness)
{
    detection_layer l = {0};
    l.type = DETECTION;
    
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.objectness = objectness;
    l.background = background;
    l.joint = joint;
    l.cost = calloc(1, sizeof(float));
    l.does_cost=1;
    int outputs = get_detection_layer_output_size(l);
    l.outputs = outputs;
    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));
    #ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*outputs);
    l.delta_gpu  = cuda_make_array(l.delta,  batch*outputs);
    #endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

void forward_detection_layer(const detection_layer l, network_state state)
{
    int in_i = 0;
    int out_i = 0;
    int locations = get_detection_layer_locations(l);
    int i,j;
    for(i = 0; i < l.batch*locations; ++i){
        int mask = (!state.truth || state.truth[out_i + (l.background || l.objectness) + l.classes + 2]);
        float scale = 1;
        if(l.joint) scale = state.input[in_i++];
        else if(l.objectness){
            l.output[out_i++] = 1-state.input[in_i++];
            scale = mask;
        }
        else if(l.background) l.output[out_i++] = scale*state.input[in_i++];

        for(j = 0; j < l.classes; ++j){
            l.output[out_i++] = scale*state.input[in_i++];
        }
        if(l.objectness){

        }else if(l.background){
            softmax_array(l.output + out_i - l.classes-l.background, l.classes+l.background, l.output + out_i - l.classes-l.background);
            activate_array(state.input+in_i, l.coords, LOGISTIC);
        }
        for(j = 0; j < l.coords; ++j){
            l.output[out_i++] = mask*state.input[in_i++];
        }
    }
    float avg_iou = 0;
    int count = 0;
    if(l.does_cost && state.train){
        *(l.cost) = 0;
        int size = get_detection_layer_output_size(l) * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (i = 0; i < l.batch*locations; ++i) {
            int classes = l.objectness+l.classes;
            int offset = i*(classes+l.coords);
            for (j = offset; j < offset+classes; ++j) {
                *(l.cost) += pow(state.truth[j] - l.output[j], 2);
                l.delta[j] =  state.truth[j] - l.output[j];
            }

            box truth;
            truth.x = state.truth[j+0]/7;
            truth.y = state.truth[j+1]/7;
            truth.w = pow(state.truth[j+2], 2);
            truth.h = pow(state.truth[j+3], 2);

            box out;
            out.x = l.output[j+0]/7;
            out.y = l.output[j+1]/7;
            out.w = pow(l.output[j+2], 2);
            out.h = pow(l.output[j+3], 2);

            if(!(truth.w*truth.h)) continue;
            float iou = box_iou(out, truth);
            avg_iou += iou;
            ++count;

            *(l.cost) += pow((1-iou), 2);
            l.delta[j+0] = 4 * (state.truth[j+0] - l.output[j+0]);
            l.delta[j+1] = 4 * (state.truth[j+1] - l.output[j+1]);
            l.delta[j+2] = 4 * (state.truth[j+2] - l.output[j+2]);
            l.delta[j+3] = 4 * (state.truth[j+3] - l.output[j+3]);
            if(l.rescore){
                for (j = offset; j < offset+classes; ++j) {
                    if(state.truth[j]) state.truth[j] = iou;
                    l.delta[j] =  state.truth[j] - l.output[j];
                }
            }
        }
        printf("Avg IOU: %f\n", avg_iou/count);
    }
}

void backward_detection_layer(const detection_layer l, network_state state)
{
    int locations = get_detection_layer_locations(l);
    int i,j;
    int in_i = 0;
    int out_i = 0;
    for(i = 0; i < l.batch*locations; ++i){
        float scale = 1;
        float latent_delta = 0;
        if(l.joint) scale = state.input[in_i++];
        else if (l.objectness)   state.delta[in_i++] += -l.delta[out_i++];
        else if (l.background) state.delta[in_i++] += scale*l.delta[out_i++];
        for(j = 0; j < l.classes; ++j){
            latent_delta += state.input[in_i]*l.delta[out_i];
            state.delta[in_i++] += scale*l.delta[out_i++];
        }

        if (l.objectness) {

        }else if (l.background) gradient_array(l.output + out_i, l.coords, LOGISTIC, l.delta + out_i);
        for(j = 0; j < l.coords; ++j){
            state.delta[in_i++] += l.delta[out_i++];
        }
        if(l.joint) state.delta[in_i-l.coords-l.classes-l.joint] += latent_delta;
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network_state state)
{
    int outputs = get_detection_layer_output_size(l);
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        truth_cpu = calloc(l.batch*outputs, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, l.batch*outputs);
    }
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_detection_layer(l, cpu_state);
    cuda_push_array(l.output_gpu, l.output, l.batch*outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*outputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_detection_layer_gpu(detection_layer l, network_state state)
{
    int outputs = get_detection_layer_output_size(l);

    float *in_cpu    = calloc(l.batch*l.inputs, sizeof(float));
    float *delta_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        truth_cpu = calloc(l.batch*outputs, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, l.batch*outputs);
    }
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.input = in_cpu;
    cpu_state.truth = truth_cpu;
    cpu_state.delta = delta_cpu;

    cuda_pull_array(state.input, in_cpu,    l.batch*l.inputs);
    cuda_pull_array(state.delta, delta_cpu, l.batch*l.inputs);
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*outputs);
    backward_detection_layer(l, cpu_state);
    cuda_push_array(state.delta, delta_cpu, l.batch*l.inputs);

    if (truth_cpu) free(truth_cpu);
    free(in_cpu);
    free(delta_cpu);
}
#endif

