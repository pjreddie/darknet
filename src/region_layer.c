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
    assert(side*side*l.coords*l.n == inputs);
    l.cost = calloc(1, sizeof(float));
    int outputs = l.n*5*side*side;
    l.outputs = outputs;
    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*inputs, sizeof(float));
    #ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*inputs);
#endif

    fprintf(stderr, "Region Layer\n");
    srand(0);

    return l;
}

void forward_region_layer(const region_layer l, network_state state)
{
    int locations = l.side*l.side;
    int i,j;
    for(i = 0; i < l.batch*locations; ++i){
        for(j = 0; j < l.n; ++j){
            int in_index =  i*l.n*l.coords + j*l.coords;
            int out_index = i*l.n*5 + j*5;

            float prob =  state.input[in_index+0];
            float x =     state.input[in_index+1];
            float y =     state.input[in_index+2];
            float w =     state.input[in_index+3];
            float h =     state.input[in_index+4];
            /*
            float min_w = state.input[in_index+5];
            float max_w = state.input[in_index+6];
            float min_h = state.input[in_index+7];
            float max_h = state.input[in_index+8];
            */

            l.output[out_index+0] = prob;
            l.output[out_index+1] = x;
            l.output[out_index+2] = y;
            l.output[out_index+3] = w;
            l.output[out_index+4] = h;

        }
    }
    if(state.train){
        float avg_iou = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (i = 0; i < l.batch*locations; ++i) {

            for(j = 0; j < l.n; ++j){
                int in_index = i*l.n*l.coords + j*l.coords;
                l.delta[in_index+0] = .1*(0-state.input[in_index+0]);
            }

            int truth_index = i*5;
            int best_index = -1;
            float best_iou = 0;
            float best_rmse = 4;

            int bg = !state.truth[truth_index];
            if(bg) continue;

            box truth = {state.truth[truth_index+1], state.truth[truth_index+2], state.truth[truth_index+3], state.truth[truth_index+4]};
            truth.x /= l.side;
            truth.y /= l.side;

            for(j = 0; j < l.n; ++j){
                int out_index = i*l.n*5 + j*5;
                box out = {l.output[out_index+1], l.output[out_index+2], l.output[out_index+3], l.output[out_index+4]};

                //printf("\n%f %f %f %f %f\n", l.output[out_index+0], out.x, out.y, out.w, out.h);

                out.x /= l.side;
                out.y /= l.side;

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
            printf("%d", best_index);
            //int out_index = i*l.n*5 + best_index*5;
            //box out = {l.output[out_index+1], l.output[out_index+2], l.output[out_index+3], l.output[out_index+4]};
            int in_index =  i*l.n*l.coords + best_index*l.coords;

            l.delta[in_index+0] = (1-state.input[in_index+0]);
            l.delta[in_index+1] = state.truth[truth_index+1] - state.input[in_index+1];
            l.delta[in_index+2] = state.truth[truth_index+2] - state.input[in_index+2];
            l.delta[in_index+3] = state.truth[truth_index+3] - state.input[in_index+3];
            l.delta[in_index+4] = state.truth[truth_index+4] - state.input[in_index+4];
            /*
            l.delta[in_index+5] = 0 - state.input[in_index+5];
            l.delta[in_index+6] = 1 - state.input[in_index+6];
            l.delta[in_index+7] = 0 - state.input[in_index+7];
            l.delta[in_index+8] = 1 - state.input[in_index+8];
            */

            /*
            float x =     state.input[in_index+1];
            float y =     state.input[in_index+2];
            float w =     state.input[in_index+3];
            float h =     state.input[in_index+4];
            float min_w = state.input[in_index+5];
            float max_w = state.input[in_index+6];
            float min_h = state.input[in_index+7];
            float max_h = state.input[in_index+8];
            */


            avg_iou += best_iou;
            ++count;
        }
        printf("\nAvg IOU: %f %d\n", avg_iou/count, count);
    }
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
    //copy_cpu(l.batch*l.inputs, l.delta, 1, state.delta, 1);
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

