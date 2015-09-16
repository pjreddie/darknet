#include "region_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

region_layer make_region_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    region_layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));
#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Region Layer\n");
    srand(0);

    return l;
}

void forward_region_layer(const region_layer l, network_state state)
{
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    for(i = 0; i < l.batch*locations; ++i){
        int index = i*((1+l.coords)*l.n + l.classes);
        if(l.softmax){
            activate_array(l.output + index, l.n*(1+l.coords), LOGISTIC);
            int offset = l.n*(1+l.coords);
            softmax_array(l.output + index + offset, l.classes,
                    l.output + index + offset);
        }
    }
    if(state.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (i = 0; i < l.batch*locations; ++i) {
            int index = i*((1+l.coords)*l.n + l.classes);
            for(j = 0; j < l.n; ++j){
                int prob_index = index + j*(1 + l.coords);
                l.delta[prob_index] = (1./l.n)*(0-l.output[prob_index]);
                if(l.softmax){
                    l.delta[prob_index] = 1./(l.n*l.side)*(0-l.output[prob_index]);
                }
                *(l.cost) += (1./l.n)*pow(l.output[prob_index], 2);
                //printf("%f\n", l.output[prob_index]);
                avg_anyobj += l.output[prob_index];
            }

            int truth_index = i*(1 + l.coords + l.classes);
            int best_index = -1;
            float best_iou = 0;
            float best_rmse = 4;

            int bg = !state.truth[truth_index];
            if(bg) {
                continue;
            }

            int class_index = index + l.n*(1+l.coords);
            for(j = 0; j < l.classes; ++j) {
                l.delta[class_index+j] = state.truth[truth_index+1+j] - l.output[class_index+j];
                *(l.cost) += pow(state.truth[truth_index+1+j] - l.output[class_index+j], 2);
                if(state.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
            }
            truth_index += l.classes + 1;
            box truth = {state.truth[truth_index+0], state.truth[truth_index+1], state.truth[truth_index+2], state.truth[truth_index+3]};
            truth.x /= l.side;
            truth.y /= l.side;

            for(j = 0; j < l.n; ++j){
                int out_index = index + j*(1+l.coords);
                box out = {l.output[out_index+1], l.output[out_index+2], l.output[out_index+3], l.output[out_index+4]};

                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt){
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }

                float iou  = box_iou(out, truth);
                float rmse = box_rmse(out, truth);
                if(best_iou > 0 || iou > 0){
                    if(iou > best_iou){
                        best_iou = iou;
                        best_index = j;
                    }
                }else{
                    if(rmse < best_rmse){
                        best_rmse = rmse;
                        best_index = j;
                    }
                }
            }
            //printf("%d", best_index);
            int in_index = index + best_index*(1+l.coords);
            *(l.cost) -= pow(l.output[in_index], 2);
            *(l.cost) += pow(1-l.output[in_index], 2);
            avg_obj += l.output[in_index];
            l.delta[in_index+0] = (1.-l.output[in_index]);
            if(l.softmax){
                l.delta[in_index+0] = 5*(1.-l.output[in_index]);
            }
            //printf("%f\n", l.output[in_index]);

            l.delta[in_index+1] = 5*(state.truth[truth_index+0] - l.output[in_index+1]);
            l.delta[in_index+2] = 5*(state.truth[truth_index+1] - l.output[in_index+2]);
            if(l.sqrt){
                l.delta[in_index+3] = 5*(sqrt(state.truth[truth_index+2]) - l.output[in_index+3]);
                l.delta[in_index+4] = 5*(sqrt(state.truth[truth_index+3]) - l.output[in_index+4]);
            }else{
                l.delta[in_index+3] = 5*(state.truth[truth_index+2] - l.output[in_index+3]);
                l.delta[in_index+4] = 5*(state.truth[truth_index+3] - l.output[in_index+4]);
            }

            *(l.cost) += pow(1-best_iou, 2);
            avg_iou += best_iou;
            ++count;
            if(l.softmax){
                gradient_array(l.output + index, l.n*(1+l.coords), LOGISTIC, l.delta + index);
            }
        }
        printf("Avg IOU: %f, Avg Cat Pred: %f, Avg Obj: %f, Avg Any: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
    }
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_region_layer_gpu(const region_layer l, network_state state)
{
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.side*l.side*(1+l.coords+l.classes);
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_region_layer(l, cpu_state);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer_gpu(region_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

