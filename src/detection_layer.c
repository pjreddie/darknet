#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
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
    l.output_gpu = cuda_make_array(0, batch*outputs);
    l.delta_gpu = cuda_make_array(0, batch*outputs);
    #endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

typedef struct{
    float dx, dy, dw, dh;
} dbox;

dbox derivative(box a, box b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

dbox dintersect(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}

dbox dunion(box a, box b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}

dbox diou(box a, box b);

void test_dunion()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
    test_dintersect();
    test_dunion();
    box a = {0, 0, 1, 1};
    box dxa= {0+.00001, 0, 1, 1};
    box dya= {0, 0+.00001, 1, 1};
    box dwa= {0, 0, 1+.00001, 1};
    box dha= {0, 0, 1, 1+.00001};

    box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    dbox d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
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
            dbox delta = diou(out, truth);

            l.delta[j+0] = 10 * delta.dx/7;
            l.delta[j+1] = 10 * delta.dy/7;
            l.delta[j+2] = 10 * delta.dw * 2 * sqrt(out.w);
            l.delta[j+3] = 10 * delta.dh * 2 * sqrt(out.h);


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
        else if (l.objectness)   state.delta[in_i++] = -l.delta[out_i++];
        else if (l.background) state.delta[in_i++] = scale*l.delta[out_i++];
        for(j = 0; j < l.classes; ++j){
            latent_delta += state.input[in_i]*l.delta[out_i];
            state.delta[in_i++] = scale*l.delta[out_i++];
        }

        if (l.objectness) {

        }else if (l.background) gradient_array(l.output + out_i, l.coords, LOGISTIC, l.delta + out_i);
        for(j = 0; j < l.coords; ++j){
            state.delta[in_i++] = l.delta[out_i++];
        }
        if(l.joint) state.delta[in_i-l.coords-l.classes-l.joint] = latent_delta;
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

    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*outputs);
    backward_detection_layer(l, cpu_state);
    cuda_push_array(state.delta, delta_cpu, l.batch*l.inputs);

    free(in_cpu);
    free(delta_cpu);
}
#endif

