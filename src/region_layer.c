#include "region_layer.h"
#include "activations.h"
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
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(5);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Region Layer\n");
    srand(0);

    return l;
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + .5)/w + x[index + 0] * biases[2*n];
    b.y = (j + .5)/h + x[index + 1] * biases[2*n + 1];
    b.w = exp(x[index + 2]) * biases[2*n];
    b.h = exp(x[index + 3]) * biases[2*n+1];
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);

    float tx = (truth.x - (i + .5)/w) / biases[2*n];
    float ty = (truth.y - (j + .5)/h) / biases[2*n + 1];
    float tw = log(truth.w / biases[2*n]);
    float th = log(truth.h / biases[2*n + 1]);

    delta[index + 0] = scale * (tx - x[index + 0]);
    delta[index + 1] = scale * (ty - x[index + 1]);
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);
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

#define LOG 0

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
                softmax(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    }
    if(!state.train) return;
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    float avg_iou = 0;
    float recall = 0;
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
                    box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
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
                        delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
                        //l.delta[index + 0] = .1 * (0 - l.output[index + 0]);
                        //l.delta[index + 1] = .1 * (0 - l.output[index + 1]);
                        //l.delta[index + 2] = .1 * (0 - l.output[index + 2]);
                        //l.delta[index + 3] = .1 * (0 - l.output[index + 3]);
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
                box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                printf("pred: (%f, %f) %f x %f\n", pred.x*l.w - i - .5, pred.y * l.h - j - .5, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_index = index;
                    best_iou = iou;
                    best_n = n;
                }
            }
            printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x * l.w - i - .5, truth.y*l.h - j - .5, truth.w, truth.h);

            float iou = delta_region_box(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
            if(iou > .5) recall += 1;
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
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            int box_index = index * (l.classes + 5);
            boxes[index].x = (predictions[box_index + 0] + col + .5) / l.w * w;
            boxes[index].y = (predictions[box_index + 1] + row + .5) / l.h * h;
            if(0){
                boxes[index].x = (logistic_activate(predictions[box_index + 0]) + col) / l.w * w;
                boxes[index].y = (logistic_activate(predictions[box_index + 1]) + row) / l.h * h;
            }
            boxes[index].w = pow(logistic_activate(predictions[box_index + 2]), (l.sqrt?2:1)) * w;
            boxes[index].h = pow(logistic_activate(predictions[box_index + 3]), (l.sqrt?2:1)) * h;
            if(1){
                boxes[index].x = ((col + .5)/l.w + predictions[box_index + 0] * .5) * w;
                boxes[index].y = ((row + .5)/l.h + predictions[box_index + 1] * .5) * h;
                boxes[index].w = (exp(predictions[box_index + 2]) * .5) * w;
                boxes[index].h = (exp(predictions[box_index + 3]) * .5) * h;
            }
            for(j = 0; j < l.classes; ++j){
                int class_index = index * (l.classes + 5) + 5;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
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

