#include "region_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int get_region_layer_locations(region_layer l)
{
    return l.inputs / (l.classes+l.coords);
}

region_layer make_region_layer(int batch, int inputs, int n, int classes, int coords, int rescore)
{
    region_layer l = {0};
    l.type = REGION;
    
    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.cost = calloc(1, sizeof(float));
    int outputs = inputs;
    l.outputs = outputs;
    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));
    #ifdef GPU
    l.output_gpu = cuda_make_array(0, batch*outputs);
    l.delta_gpu = cuda_make_array(0, batch*outputs);
    #endif

    fprintf(stderr, "Region Layer\n");
    srand(0);

    return l;
}

void forward_region_layer(const region_layer l, network_state state)
{
    int locations = get_region_layer_locations(l);
    int i,j;
    for(i = 0; i < l.batch*locations; ++i){
        int index = i*(l.classes + l.coords);
        int mask = (!state.truth || !state.truth[index]);

        for(j = 0; j < l.classes; ++j){
            l.output[index+j] = state.input[index+j];
        }

        softmax_array(l.output + index, l.classes, l.output + index);
        index += l.classes;

        for(j = 0; j < l.coords; ++j){
            l.output[index+j] = mask*state.input[index+j];
        }
    }
    if(state.train){
        float avg_iou = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.outputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (i = 0; i < l.batch*locations; ++i) {
            int offset = i*(l.classes+l.coords);
            int bg = state.truth[offset];
            for (j = offset; j < offset+l.classes; ++j) {
                //*(l.cost) += pow(state.truth[j] - l.output[j], 2);
                //l.delta[j] =  state.truth[j] - l.output[j];
            }

            box anchor = {0,0,.5,.5};
            box truth_code = {state.truth[j+0], state.truth[j+1], state.truth[j+2], state.truth[j+3]};
            box out_code =   {l.output[j+0], l.output[j+1], l.output[j+2], l.output[j+3]};
            box out = decode_box(out_code, anchor);
            box truth = decode_box(truth_code, anchor);

            if(bg) continue;
            //printf("Box:       %f %f %f %f\n", truth.x, truth.y, truth.w, truth.h);
            //printf("Code:      %f %f %f %f\n", truth_code.x, truth_code.y, truth_code.w, truth_code.h);
            //printf("Pred     : %f %f %f %f\n", out.x, out.y, out.w, out.h);
            // printf("Pred Code: %f %f %f %f\n", out_code.x, out_code.y, out_code.w, out_code.h);
            float iou = box_iou(out, truth);
            avg_iou += iou;
            ++count;

            /*
             *(l.cost) += pow((1-iou), 2);
             l.delta[j+0] = (state.truth[j+0] - l.output[j+0]);
             l.delta[j+1] = (state.truth[j+1] - l.output[j+1]);
             l.delta[j+2] = (state.truth[j+2] - l.output[j+2]);
             l.delta[j+3] = (state.truth[j+3] - l.output[j+3]);
             */

            for (j = offset+l.classes; j < offset+l.classes+l.coords; ++j) {
                //*(l.cost) += pow(state.truth[j] - l.output[j], 2);
                //l.delta[j] =  state.truth[j] - l.output[j];
                float diff = state.truth[j] - l.output[j];
                if (fabs(diff) < 1){
                    l.delta[j] = diff;
                    *(l.cost) += .5*pow(state.truth[j] - l.output[j], 2);
                } else {
                    l.delta[j] = (diff > 0) ? 1 : -1;
                    *(l.cost) += fabs(diff) - .5;
                }
                //l.delta[j] = state.truth[j] - l.output[j];
            }

            /*
               if(l.rescore){
               for (j = offset; j < offset+l.classes; ++j) {
               if(state.truth[j]) state.truth[j] = iou;
               l.delta[j] =  state.truth[j] - l.output[j];
               }
               }
             */
        }
        printf("Avg IOU: %f\n", avg_iou/count);
    }
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_cpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}

#ifdef GPU

void forward_region_layer_gpu(const region_layer l, network_state state)
{
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        truth_cpu = calloc(l.batch*l.outputs, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, l.batch*l.outputs);
    }
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_region_layer(l, cpu_state);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer_gpu(region_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

