// Gaussian YOLOv3 implementation
// Author: Jiwoong Choi
// ICCV 2019 Paper: http://openaccess.thecvf.com/content_ICCV_2019/html/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.html
// arxiv.org: https://arxiv.org/abs/1904.04620v2
// source code: https://github.com/jwchoi384/Gaussian_YOLOv3

#include "gaussian_yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.141592
#endif

layer make_gaussian_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = GAUSSIAN_YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 8 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)calloc(1, sizeof(float));
    l.biases = (float*)calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = (int*)calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 8 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes*(4 + 1);
    l.delta = (float*)calloc(batch*l.outputs, sizeof(float));
    l.output = (float*)calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_gaussian_yolo_layer;
    l.backward = backward_gaussian_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_gaussian_yolo_layer_gpu;
    l.backward_gpu = backward_gaussian_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    /*
    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs * sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)calloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs * sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)calloc(batch * l.outputs, sizeof(float));
    }
    */
#endif

    fprintf(stderr, "Gaussian_yolo\n");
    srand(time(0));

    return l;
}

void resize_gaussian_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 8 + 1);
    l->inputs = l->outputs;

    l->output = (float *)realloc(l->output, l->batch*l->outputs * sizeof(float));
    l->delta = (float *)realloc(l->delta, l->batch*l->outputs * sizeof(float));

    //if (!l->output_pinned) l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
    //if (!l->delta_pinned) l->delta = (float*)realloc(l->delta, l->batch*l->outputs * sizeof(float));

#ifdef GPU
    /*
    if (l->output_pinned) {
        cudaFreeHost(l->output);
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)realloc(l->output, l->batch * l->outputs * sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        cudaFreeHost(l->delta);
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)realloc(l->delta, l->batch * l->outputs * sizeof(float));
            l->delta_pinned = 0;
        }
    }
    */

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_gaussian_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 2*stride]) / lh;
    b.w = exp(x[index + 4*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 6*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_gaussian_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_gaussian_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    float sigma_const = 0.3;
    float epsi = pow(10,-9);

    float in_exp_x = (tx - x[index + 0*stride])/x[index+1*stride];
    float in_exp_x_2 = pow(in_exp_x, 2);
    float normal_dist_x = exp(in_exp_x_2*(-1./2.))/(sqrt(M_PI * 2.0)*(x[index+1*stride]+sigma_const));

    float in_exp_y = (ty - x[index + 2*stride])/x[index+3*stride];
    float in_exp_y_2 = pow(in_exp_y, 2);
    float normal_dist_y = exp(in_exp_y_2*(-1./2.))/(sqrt(M_PI * 2.0)*(x[index+3*stride]+sigma_const));

    float in_exp_w = (tw - x[index + 4*stride])/x[index+5*stride];
    float in_exp_w_2 = pow(in_exp_w, 2);
    float normal_dist_w = exp(in_exp_w_2*(-1./2.))/(sqrt(M_PI * 2.0)*(x[index+5*stride]+sigma_const));

    float in_exp_h = (th - x[index + 6*stride])/x[index+7*stride];
    float in_exp_h_2 = pow(in_exp_h, 2);
    float normal_dist_h = exp(in_exp_h_2*(-1./2.))/(sqrt(M_PI * 2.0)*(x[index+7*stride]+sigma_const));

    float temp_x = (1./2.) * 1./(normal_dist_x+epsi) * normal_dist_x * scale;
    float temp_y = (1./2.) * 1./(normal_dist_y+epsi) * normal_dist_y * scale;
    float temp_w = (1./2.) * 1./(normal_dist_w+epsi) * normal_dist_w * scale;
    float temp_h = (1./2.) * 1./(normal_dist_h+epsi) * normal_dist_h * scale;

    delta[index + 0*stride] = temp_x * in_exp_x  * (1./x[index+1*stride]);
    delta[index + 2*stride] = temp_y * in_exp_y  * (1./x[index+3*stride]);
    delta[index + 4*stride] = temp_w * in_exp_w  * (1./x[index+5*stride]);
    delta[index + 6*stride] = temp_h * in_exp_h  * (1./x[index+7*stride]);

    delta[index + 1*stride] = temp_x * (in_exp_x_2/x[index+1*stride] - 1./(x[index+1*stride]+sigma_const));
    delta[index + 3*stride] = temp_y * (in_exp_y_2/x[index+3*stride] - 1./(x[index+3*stride]+sigma_const));
    delta[index + 5*stride] = temp_w * (in_exp_w_2/x[index+5*stride] - 1./(x[index+5*stride]+sigma_const));
    delta[index + 7*stride] = temp_h * (in_exp_h_2/x[index+7*stride] - 1./(x[index+7*stride]+sigma_const));
    return iou;
}


void delta_gaussian_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*class_id] = 1 - output[index + stride*class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class_id)?1 : 0) - output[index + stride*n];
        if(n == class_id && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_gaussian_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(8+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_gaussian_yolo_layer(const layer l, network_state state)
{
    int i,j,b,t,n;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            // x : mu, sigma
            int index = entry_gaussian_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            scal_add_cpu(l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);    // scale x
            // y : mu, sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 2);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            scal_add_cpu(l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);    // scale y
            // w : sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 5);
            activate_array(l.output + index, l.w*l.h, LOGISTIC);
            // h : sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 7);
            activate_array(l.output + index, l.w*l.h, LOGISTIC);
            // objectness & class
            index = entry_gaussian_index(l, b, n*l.w*l.h, 8);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!state.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_gaussian_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_gaussian_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_gaussian_index(l, b, n*l.w*l.h + j*l.w + i, 8);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int class_id = state.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class_id = l.map[class_id];
                        int class_index = entry_gaussian_index(l, b, n*l.w*l.h + j*l.w + i, 9);
                        delta_gaussian_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0);
                        box truth = float_to_box_stride(state.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_gaussian_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/ state.net.w;
                pred.h = l.biases[2*n+1]/ state.net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                int box_index = entry_gaussian_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_gaussian_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_gaussian_index(l, b, mask_n*l.w*l.h + j*l.w + i, 8);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class_id = l.map[class_id];
                int class_index = entry_gaussian_index(l, b, mask_n*l.w*l.h + j*l.w + i, 9);
                delta_gaussian_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_gaussian_yolo_layer(const layer l, network_state state)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void correct_gaussian_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    /*
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    */
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int gaussian_yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_gaussian_index(l, 0, n*l.w*l.h + i, 8);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

/*
void avg_flipped_gaussian_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 8 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}
*/

int get_gaussian_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    int i,j,n;
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_gaussian_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_gaussian_index(l, 0, n*l.w*l.h + i, 8);
            float objectness = predictions[obj_index];
            if (objectness <= thresh) continue;    // incorrect behavior for Nan values

            if (objectness > thresh) {
                int box_index = entry_gaussian_index(l, 0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_gaussian_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;

                dets[count].uc[0] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 1)]; // tx uncertainty
                dets[count].uc[1] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 3)]; // ty uncertainty
                dets[count].uc[2] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 5)]; // tw uncertainty
                dets[count].uc[3] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 7)]; // th uncertainty

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_gaussian_index(l, 0, n*l.w*l.h + i, 9 + j);
                    float uc_aver = (dets[count].uc[0] + dets[count].uc[1] + dets[count].uc[2] + dets[count].uc[3]) / 4.0;
                    float prob = objectness*predictions[class_index] * (1.0 - uc_aver);
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_gaussian_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_gaussian_yolo_layer_gpu(const layer l, network_state state)
{
    copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b)
    {
        for(n = 0; n < l.n; ++n)
        {
            // x : mu, sigma
            int index = entry_gaussian_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            scal_add_ongpu(l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale x
            // y : mu, sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 2);
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            scal_add_ongpu(l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale y
            // w : sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 5);
            activate_array_ongpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
            // h : sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 7);
            activate_array_ongpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
            // objectness & class
            index = entry_gaussian_index(l, b, n*l.w*l.h, 8);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }

    if (!state.train || l.onlyforward) {
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs * sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_gaussian_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_gaussian_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}
#endif
