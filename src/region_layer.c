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

region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    region_layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(5);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Region Layer\n");
    srand(0);

    return l;
}

box get_region_box2(float *x, int index, int i, int j, int w, int h)
{
    float aspect = exp(x[index+0]);
    float scale  = logistic_activate(x[index+1]);
    float move_x = x[index+2];
    float move_y = x[index+3];

    box b;
    b.w = sqrt(scale * aspect);
    b.h = b.w * 1./aspect;
    b.x = move_x * b.w + (i + .5)/w;
    b.y = move_y * b.h + (j + .5)/h;
    return b;
}

float delta_region_box2(box truth, float *output, int index, int i, int j, int w, int h, float *delta)
{
    box pred = get_region_box2(output, index, i, j, w, h);
    float iou = box_iou(pred, truth);
    float true_aspect = truth.w/truth.h;
    float true_scale = truth.w*truth.h;

    float true_dx = (truth.x - (i+.5)/w) / truth.w;
    float true_dy = (truth.y - (j+.5)/h) / truth.h;
    delta[index + 0] = (true_aspect - exp(output[index + 0])) * exp(output[index + 0]);
    delta[index + 1] = (true_scale - logistic_activate(output[index + 1])) * logistic_gradient(logistic_activate(output[index + 1]));
    delta[index + 2] = true_dx - output[index + 2];
    delta[index + 3] = true_dy - output[index + 3];
    return iou;
}

box get_region_box(float *x, int index, int i, int j, int w, int h, int adjust, int logistic)
{
    box b;
    b.x = (x[index + 0] + i + .5)/w;
    b.y = (x[index + 1] + j + .5)/h;
    b.w = x[index + 2];
    b.h = x[index + 3];
    if(logistic){
        b.w = logistic_activate(x[index + 2]);
        b.h = logistic_activate(x[index + 3]);
    }
    if(adjust && b.w < .01) b.w = .01;
    if(adjust && b.h < .01) b.h = .01;
    return b;
}

float delta_region_box(box truth, float *output, int index, int i, int j, int w, int h, float *delta, int logistic, float scale)
{
    box pred = get_region_box(output, index, i, j, w, h, 0, logistic);
    float iou = box_iou(pred, truth);

    delta[index + 0] = scale * (truth.x - pred.x);
    delta[index + 1] = scale * (truth.y - pred.y);
    delta[index + 2] = scale * ((truth.w - pred.w)*(logistic ? logistic_gradient(pred.w) : 1));
    delta[index + 3] = scale * ((truth.h - pred.h)*(logistic ? logistic_gradient(pred.h) : 1));
    return iou;
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

#define LOG 1

void forward_region_layer(const region_layer l, network_state state)
{
    int i,j,b,t,n;
    int size = l.coords + l.classes + 1;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.h*l.w*l.n; ++i){
            int index = size*i + b*l.outputs;
            l.output[index + 4] = logistic_activate(l.output[index + 4]);
            if(l.softmax){
                softmax_array(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    }
    if(!state.train) return;
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    float avg_iou = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                    box pred = get_region_box(l.output, index, i, j, l.w, l.h, 1, LOG);
                    float best_iou = 0;
                    for(t = 0; t < 30; ++t){
                        box truth = float_to_box(state.truth + t*5 + b*l.truths);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) best_iou = iou;
                    }
                    avg_anyobj += l.output[index + 4];
                    l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    if(best_iou > .5) l.delta[index + 4] = 0;

                    if(*(state.net.seen) < 6400){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = .5;
                        truth.h = .5;
                        delta_region_box(truth, l.output, index, i, j, l.w, l.h, l.delta, LOG, 1);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(state.truth + t*5 + b*l.truths);
            int class = state.truth[t*5 + b*l.truths + 4];
            if(!truth.x) break;
            float best_iou = 0;
            int best_index = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){
                int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                box pred = get_region_box(l.output, index, i, j, l.w, l.h, 1, LOG);
                printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_index = index;
                    best_iou = iou;
                    best_n = n;
                }
            }
            printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

            float iou = delta_region_box(truth, l.output, best_index, i, j, l.w, l.h, l.delta, LOG, l.coord_scale);
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            avg_obj += l.output[best_index + 4];
            l.delta[best_index + 4] = l.object_scale * (1 - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            if (l.rescore) {
                l.delta[best_index + 4] = l.object_scale * (iou - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            }
            //printf("%f\n", l.delta[best_index+1]);
            /*
               if(isnan(l.delta[best_index+1])){
               printf("%f\n", true_scale);
               printf("%f\n", l.output[best_index + 1]);
               printf("%f\n", truth.w);
               printf("%f\n", truth.h);
               error("bad");
               }
             */
            for(n = 0; n < l.classes; ++n){
                l.delta[best_index + 5 + n] = l.class_scale * (((n == class)?1 : 0) - l.output[best_index + 5 + n]);
                if(n == class) avg_cat += l.output[best_index + 5 + n];
            }
            /*
               if(0){
               printf("truth: %f %f %f %f\n", truth.x, truth.y, truth.w, truth.h);
               printf("pred: %f %f %f %f\n\n", pred.x, pred.y, pred.w, pred.h);
               float aspect = exp(true_aspect);
               float scale  = logistic_activate(true_scale);
               float move_x = true_dx;
               float move_y = true_dy;

               box b;
               b.w = sqrt(scale * aspect);
               b.h = b.w * 1./aspect;
               b.x = move_x * b.w + (i + .5)/l.w;
               b.y = move_y * b.h + (j + .5)/l.h;
               printf("%f %f\n", b.x, truth.x);
               printf("%f %f\n", b.y, truth.y);
               printf("%f %f\n", b.w, truth.w);
               printf("%f %f\n", b.h, truth.h);
            //printf("%f\n", box_iou(b, truth));
            }
             */
            ++count;
        }
    }
    printf("\n");
    reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), count);
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_region_layer_gpu(const region_layer l, network_state state)
{
    /*
       if(!state.train){
       copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
       return;
       }
     */

    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state = state;
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
    axpy_ongpu(l.batch*l.outputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

