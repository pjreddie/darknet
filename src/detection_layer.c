#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int get_detection_layer_locations(detection_layer layer)
{
    return layer.inputs / (layer.classes+layer.coords+layer.rescore+layer.background);
}

int get_detection_layer_output_size(detection_layer layer)
{
    return get_detection_layer_locations(layer)*(layer.background + layer.classes + layer.coords);
}

detection_layer *make_detection_layer(int batch, int inputs, int classes, int coords, int rescore, int background, int nuisance)
{
    detection_layer *layer = calloc(1, sizeof(detection_layer));
    
    layer->batch = batch;
    layer->inputs = inputs;
    layer->classes = classes;
    layer->coords = coords;
    layer->rescore = rescore;
    layer->nuisance = nuisance;
    layer->cost = calloc(1, sizeof(float));
    layer->does_cost=1;
    layer->background = background;
    int outputs = get_detection_layer_output_size(*layer);
    layer->output = calloc(batch*outputs, sizeof(float));
    layer->delta = calloc(batch*outputs, sizeof(float));
    #ifdef GPU
    layer->output_gpu = cuda_make_array(0, batch*outputs);
    layer->delta_gpu = cuda_make_array(0, batch*outputs);
    #endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return layer;
}

void dark_zone(detection_layer layer, int class, int start, network_state state)
{
    int index = start+layer.background+class;
    int size = layer.classes+layer.coords+layer.background;
    int location = (index%(7*7*size)) / size ;
    int r = location / 7;
    int c = location % 7;
    int dr, dc;
    for(dr = -1; dr <= 1; ++dr){
        for(dc = -1; dc <= 1; ++dc){
            if(!(dr || dc)) continue;
            if((r + dr) > 6 || (r + dr) < 0) continue;
            if((c + dc) > 6 || (c + dc) < 0) continue;
            int di = (dr*7 + dc) * size;
            if(state.truth[index+di]) continue;
            layer.output[index + di] = 0;
            //if(!state.truth[start+di]) continue;
            //layer.output[start + di] = 1;
        }
    }
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
    if(h < 0 || w < 0){
        di.dx = dover.dx;
        di.dy = dover.dy;
    }
    return di;
}

dbox dunion(box a, box b)
{
    dbox du = {0,0,0,0};;
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w > 0 && h > 0){
        dbox di = dintersect(a, b);
        du.dw = h - di.dw;
        du.dh = w - di.dw;
        du.dx = -di.dx;
        du.dy = -di.dy;
    }
    return du;
}

dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};
    if(i < 0) {
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

void test_box()
{
    box a = {1, 1, 1, 1};
    box b = {0, 0, .5, .2};
    int count = 0;
    while(count++ < 300){
        dbox d = diou(a, b);
        printf("%f %f %f %f\n", a.x, a.y, a.w, a.h);
        a.x += .1*d.dx;
        a.w += .1*d.dw;
        a.y += .1*d.dy;
        a.h += .1*d.dh;
        printf("inter: %f\n", box_intersection(a, b));
        printf("union: %f\n", box_union(a, b));
        printf("IOU: %f\n", box_iou(a, b));
        if(d.dx==0 && d.dw==0 && d.dy==0 && d.dh==0) {
            printf("break!!!\n");
            break;
        }
    }
}

void forward_detection_layer(const detection_layer layer, network_state state)
{
    int in_i = 0;
    int out_i = 0;
    int locations = get_detection_layer_locations(layer);
    int i,j;
    for(i = 0; i < layer.batch*locations; ++i){
        int mask = (!state.truth || state.truth[out_i + layer.background + layer.classes + 2]);
        float scale = 1;
        if(layer.rescore) scale = state.input[in_i++];
        else if(layer.nuisance){
            layer.output[out_i++] = 1-state.input[in_i++];
            scale = mask;
        }
        else if(layer.background) layer.output[out_i++] = scale*state.input[in_i++];

        for(j = 0; j < layer.classes; ++j){
            layer.output[out_i++] = scale*state.input[in_i++];
        }
        if(layer.nuisance){
            
        }else if(layer.background){
            softmax_array(layer.output + out_i - layer.classes-layer.background, layer.classes+layer.background, layer.output + out_i - layer.classes-layer.background);
            activate_array(state.input+in_i, layer.coords, LOGISTIC);
        }
        for(j = 0; j < layer.coords; ++j){
            layer.output[out_i++] = mask*state.input[in_i++];
        }
    }
    if(layer.does_cost){
        *(layer.cost) = 0;
        int size = get_detection_layer_output_size(layer) * layer.batch;
        memset(layer.delta, 0, size * sizeof(float));
        for(i = 0; i < layer.batch*locations; ++i){
            int classes = layer.nuisance+layer.classes;
            int offset = i*(classes+layer.coords);
            for(j = offset; j < offset+classes; ++j){
                *(layer.cost) += pow(state.truth[j] - layer.output[j], 2);
                layer.delta[j] =  state.truth[j] - layer.output[j];
            }
            box truth;
            truth.x = state.truth[j+0];
            truth.y = state.truth[j+1];
            truth.w = state.truth[j+2];
            truth.h = state.truth[j+3];
            box out;
            out.x = layer.output[j+0];
            out.y = layer.output[j+1];
            out.w = layer.output[j+2];
            out.h = layer.output[j+3];
            if(!(truth.w*truth.h)) continue;
            float iou = box_iou(truth, out);
            //printf("iou: %f\n", iou);
            *(layer.cost) += pow((1-iou), 2);
            dbox d = diou(out, truth);
            layer.delta[j+0] = d.dx;
            layer.delta[j+1] = d.dy;
            layer.delta[j+2] = d.dw;
            layer.delta[j+3] = d.dh;
        }
    }
    /*
       int count = 0;
       for(i = 0; i < layer.batch*locations; ++i){
       for(j = 0; j < layer.classes+layer.background; ++j){
       printf("%f, ", layer.output[count++]);
       }
       printf("\n");
       for(j = 0; j < layer.coords; ++j){
       printf("%f, ", layer.output[count++]);
       }
       printf("\n");
       }
     */
    /*
       if(layer.background || 1){
       for(i = 0; i < layer.batch*locations; ++i){
       int index = i*(layer.classes+layer.coords+layer.background);
       for(j= 0; j < layer.classes; ++j){
       if(state.truth[index+j+layer.background]){
//dark_zone(layer, j, index, state);
}
}
}
}
     */
}

void backward_detection_layer(const detection_layer layer, network_state state)
{
    int locations = get_detection_layer_locations(layer);
    int i,j;
    int in_i = 0;
    int out_i = 0;
    for(i = 0; i < layer.batch*locations; ++i){
        float scale = 1;
        float latent_delta = 0;
        if(layer.rescore) scale = state.input[in_i++];
        else if (layer.nuisance)   state.delta[in_i++] = -layer.delta[out_i++];
        else if (layer.background) state.delta[in_i++] = scale*layer.delta[out_i++];
        for(j = 0; j < layer.classes; ++j){
            latent_delta += state.input[in_i]*layer.delta[out_i];
            state.delta[in_i++] = scale*layer.delta[out_i++];
        }

        if (layer.nuisance) {

        }else if (layer.background) gradient_array(layer.output + out_i, layer.coords, LOGISTIC, layer.delta + out_i);
        for(j = 0; j < layer.coords; ++j){
            state.delta[in_i++] = layer.delta[out_i++];
        }
        if(layer.rescore) state.delta[in_i-layer.coords-layer.classes-layer.rescore-layer.background] = latent_delta;
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer layer, network_state state)
{
    int outputs = get_detection_layer_output_size(layer);
    float *in_cpu = calloc(layer.batch*layer.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        truth_cpu = calloc(layer.batch*outputs, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, layer.batch*outputs);
    }
    cuda_pull_array(state.input, in_cpu, layer.batch*layer.inputs);
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_detection_layer(layer, cpu_state);
    cuda_push_array(layer.output_gpu, layer.output, layer.batch*outputs);
    cuda_push_array(layer.delta_gpu, layer.delta, layer.batch*outputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_detection_layer_gpu(detection_layer layer, network_state state)
{
    int outputs = get_detection_layer_output_size(layer);

    float *in_cpu =    calloc(layer.batch*layer.inputs, sizeof(float));
    float *delta_cpu = calloc(layer.batch*layer.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        truth_cpu = calloc(layer.batch*outputs, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, layer.batch*outputs);
    }
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.input = in_cpu;
    cpu_state.truth = truth_cpu;
    cpu_state.delta = delta_cpu;

    cuda_pull_array(state.input, in_cpu, layer.batch*layer.inputs);
    cuda_pull_array(layer.delta_gpu, layer.delta, layer.batch*outputs);
    backward_detection_layer(layer, cpu_state);
    cuda_push_array(state.delta, delta_cpu, layer.batch*layer.inputs);

    free(in_cpu);
    free(delta_cpu);
}
#endif

