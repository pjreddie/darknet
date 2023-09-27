#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)xcalloc(1, sizeof(float));
    l.biases = (float*)xcalloc(total * 2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = (int*)xcalloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truth_size = 4 + 2;
    l.truths = l.max_boxes*l.truth_size;    // 90*(4 + 1);
    l.labels = (int*)xcalloc(batch * l.w*l.h*l.n, sizeof(int));
    for (i = 0; i < batch * l.w*l.h*l.n; ++i) l.labels[i] = -1;
    l.class_ids = (int*)xcalloc(batch * l.w*l.h*l.n, sizeof(int));
    for (i = 0; i < batch * l.w*l.h*l.n; ++i) l.class_ids[i] = -1;

    l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.output_avg_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }
#endif

    fprintf(stderr, "yolo\n");
    srand(time(0));

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    if (l->embedding_output) l->embedding_output = (float*)xrealloc(l->output, l->batch * l->embedding_size * l->n * l->h * l->w * sizeof(float));
    if (l->labels) l->labels = (int*)xrealloc(l->labels, l->batch * l->n * l->h * l->w * sizeof(int));
    if (l->class_ids) l->class_ids = (int*)xrealloc(l->class_ids, l->batch * l->n * l->h * l->w * sizeof(int));

    if (!l->output_pinned) l->output = (float*)xrealloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)xrealloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        CHECK_CUDA(cudaFreeHost(l->output));
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        CHECK_CUDA(cudaFreeHost(l->delta));
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->output_avg_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
    l->output_avg_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, int new_coords)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
    // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
    // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    if (new_coords) {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
        b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
    }
    else {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    }
    return b;
}

static inline float fix_nan_inf(float val)
{
    if (isnan(val) || isinf(val)) val = 0;
    return val;
}

static inline float clip_value(float val, const float max_val)
{
    if (val > max_val) {
        //printf("\n val = %f > max_val = %f \n", val, max_val);
        val = max_val;
    }
    else if (val < -max_val) {
        //printf("\n val = %f < -max_val = %f \n", val, -max_val);
        val = -max_val;
    }
    return val;
}

ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss, int accumulate, float max_delta, int *rewritten_bbox, int new_coords)
{
    if (delta[index + 0 * stride] || delta[index + 1 * stride] || delta[index + 2 * stride] || delta[index + 3 * stride]) {
        (*rewritten_bbox)++;
    }

    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);
    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    all_ious.diou = box_diou(pred, truth);
    all_ious.ciou = box_ciou(pred, truth);
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE)    // old loss
    {
        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2 * n]);
        float th = log(truth.h*h / biases[2 * n + 1]);

        if (new_coords) {
            //tx = (truth.x*lw - i + 0.5) / 2;
            //ty = (truth.y*lh - j + 0.5) / 2;
            tw = sqrt(truth.w*w / (4 * biases[2 * n]));
            th = sqrt(truth.h*h / (4 * biases[2 * n + 1]));
        }

        //printf(" tx = %f, ty = %f, tw = %f, th = %f \n", tx, ty, tw, th);
        //printf(" x = %f, y = %f, w = %f, h = %f \n", x[index + 0 * stride], x[index + 1 * stride], x[index + 2 * stride], x[index + 3 * stride]);

        // accumulate delta
        delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;
        delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;
        delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;
        delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;
    }
    else {
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        //float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        //float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;


        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        if (new_coords) {
            //dw *= 8 * x[index + 2 * stride];
            //dh *= 8 * x[index + 3 * stride];
            //dw *= 8 * x[index + 2 * stride] * biases[2 * n] / w;
            //dh *= 8 * x[index + 3 * stride] * biases[2 * n + 1] / h;

            //float grad_w = 8 * exp(-x[index + 2 * stride]) / pow(exp(-x[index + 2 * stride]) + 1, 3);
            //float grad_h = 8 * exp(-x[index + 3 * stride]) / pow(exp(-x[index + 3 * stride]) + 1, 3);
            //dw *= grad_w;
            //dh *= grad_h;
        }
        else {
            dw *= exp(x[index + 2 * stride]);
            dh *= exp(x[index + 3 * stride]);
        }


        //dw *= exp(x[index + 2 * stride]);
        //dh *= exp(x[index + 3 * stride]);

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;


        dx = fix_nan_inf(dx);
        dy = fix_nan_inf(dy);
        dw = fix_nan_inf(dw);
        dh = fix_nan_inf(dh);

        if (max_delta != FLT_MAX) {
            dx = clip_value(dx, max_delta);
            dy = clip_value(dy, max_delta);
            dw = clip_value(dw, max_delta);
            dh = clip_value(dh, max_delta);
        }


        if (!accumulate) {
            delta[index + 0 * stride] = 0;
            delta[index + 1 * stride] = 0;
            delta[index + 2 * stride] = 0;
            delta[index + 3 * stride] = 0;
        }

        // accumulate delta
        delta[index + 0 * stride] += dx;
        delta[index + 1 * stride] += dy;
        delta[index + 2 * stride] += dw;
        delta[index + 3 * stride] += dh;
    }

    return all_ious;
}

void averages_yolo_deltas(int class_index, int box_index, int stride, int classes, float *delta)
{

    int classes_in_one_box = 0;
    int c;
    for (c = 0; c < classes; ++c) {
        if (delta[class_index + stride*c] > 0) classes_in_one_box++;
    }

    if (classes_in_one_box > 0) {
        delta[box_index + 0 * stride] /= classes_in_one_box;
        delta[box_index + 1 * stride] /= classes_in_one_box;
        delta[box_index + 2 * stride] /= classes_in_one_box;
        delta[box_index + 3 * stride] /= classes_in_one_box;
    }
}

void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers, float cls_normalizer)
{
    int n;
    if (delta[index + stride*class_id]){
        float y_true = 1;
        if(label_smooth_eps) y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
        float result_delta = y_true - output[index + stride*class_id];
        if(!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*class_id] = result_delta;
        //delta[index + stride*class_id] = 1 - output[index + stride*class_id];

        if (classes_multipliers) delta[index + stride*class_id] *= classes_multipliers[class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            float y_true = ((n == class_id) ? 1 : 0);
            if (label_smooth_eps) y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
            float result_delta = y_true - output[index + stride*n];
            if (!isnan(result_delta) && !isinf(result_delta)) delta[index + stride*n] = result_delta;

            if (classes_multipliers && n == class_id) delta[index + stride*class_id] *= classes_multipliers[class_id] * cls_normalizer;
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

int compare_yolo_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh)
{
    int j;
    for (j = 0; j < classes; ++j) {
        //float prob = objectness * output[class_index + stride*j];
        float prob = output[class_index + stride*j];
        if (prob > conf_thresh) {
            return 1;
        }
    }
    return 0;
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

typedef struct train_yolo_args {
    layer l;
    network_state state;
    int b;

    float tot_iou;
    float tot_giou_loss;
    float tot_iou_loss;
    int count;
    int class_count;
} train_yolo_args;

void *process_batch(void* ptr)
{
    {
        train_yolo_args *args = (train_yolo_args*)ptr;
        const layer l = args->l;
        network_state state = args->state;
        int b = args->b;

        int i, j, t, n;

        //printf(" b = %d \n", b, b);

        //float tot_iou = 0;
        float tot_giou = 0;
        float tot_diou = 0;
        float tot_ciou = 0;
        //float tot_iou_loss = 0;
        //float tot_giou_loss = 0;
        float tot_diou_loss = 0;
        float tot_ciou_loss = 0;
        float recall = 0;
        float recall75 = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        //int count = 0;
        //int class_count = 0;

        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    const int class_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4 + 1);
                    const int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
                    const int box_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
                    const int stride = l.w * l.h;
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.new_coords);
                    float best_match_iou = 0;
                    int best_match_t = 0;
                    float best_iou = 0;
                    int best_t = 0;
                    for (t = 0; t < l.max_boxes; ++t) {
                        box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
                        if (!truth.x) break;  // continue;
                        int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                        if (class_id >= l.classes || class_id < 0) {
                            printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                            printf("\n truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n", truth.x, truth.y, truth.w, truth.h, class_id);
                            continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value
                        }

                        float objectness = l.output[obj_index];
                        if (isnan(objectness) || isinf(objectness)) l.output[obj_index] = 0;
                        int class_id_match = compare_yolo_class(l.output, l.classes, class_index, l.w * l.h, objectness, class_id, 0.25f);

                        float iou = box_iou(pred, truth);
                        if (iou > best_match_iou && class_id_match == 1) {
                            best_match_iou = iou;
                            best_match_t = t;
                        }
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }

                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.obj_normalizer * (0 - l.output[obj_index]);
                    if (best_match_iou > l.ignore_thresh) {
                        if (l.objectness_smooth) {
                            const float delta_obj = l.obj_normalizer * (best_match_iou - l.output[obj_index]);
                            if (delta_obj > l.delta[obj_index]) l.delta[obj_index] = delta_obj;

                        }
                        else l.delta[obj_index] = 0;
                    }
                    else if (state.net.adversarial) {
                        int stride = l.w * l.h;
                        float scale = pred.w * pred.h;
                        if (scale > 0) scale = sqrt(scale);
                        l.delta[obj_index] = scale * l.obj_normalizer * (0 - l.output[obj_index]);
                        int cl_id;
                        int found_object = 0;
                        for (cl_id = 0; cl_id < l.classes; ++cl_id) {
                            if (l.output[class_index + stride * cl_id] * l.output[obj_index] > 0.25) {
                                l.delta[class_index + stride * cl_id] = scale * (0 - l.output[class_index + stride * cl_id]);
                                found_object = 1;
                            }
                        }
                        if (found_object) {
                            // don't use this loop for adversarial attack drawing
                            for (cl_id = 0; cl_id < l.classes; ++cl_id)
                                if (l.output[class_index + stride * cl_id] * l.output[obj_index] < 0.25)
                                    l.delta[class_index + stride * cl_id] = scale * (1 - l.output[class_index + stride * cl_id]);

                            l.delta[box_index + 0 * stride] += scale * (0 - l.output[box_index + 0 * stride]);
                            l.delta[box_index + 1 * stride] += scale * (0 - l.output[box_index + 1 * stride]);
                            l.delta[box_index + 2 * stride] += scale * (0 - l.output[box_index + 2 * stride]);
                            l.delta[box_index + 3 * stride] += scale * (0 - l.output[box_index + 3 * stride]);
                        }
                    }
                    if (best_iou > l.truth_thresh) {
                        const float iou_multiplier = best_iou * best_iou;// (best_iou - l.truth_thresh) / (1.0 - l.truth_thresh);
                        if (l.objectness_smooth) l.delta[obj_index] = l.obj_normalizer * (iou_multiplier - l.output[obj_index]);
                        else l.delta[obj_index] = l.obj_normalizer * (1 - l.output[obj_index]);
                        //l.delta[obj_index] = l.obj_normalizer * (1 - l.output[obj_index]);

                        int class_id = state.truth[best_t * l.truth_size + b * l.truths + 4];
                        if (l.map) class_id = l.map[class_id];
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, 0, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        if (l.objectness_smooth) l.delta[class_index + stride * class_id] = class_multiplier * (iou_multiplier - l.output[class_index + stride * class_id]);
                        box truth = float_to_box_stride(state.truth + best_t * l.truth_size + b * l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                        (*state.net.total_bbox)++;
                    }
                }
            }
        }
        for (t = 0; t < l.max_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
            if (!truth.x) break;  // continue;
            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                char buff[256];
                printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
                sprintf(buff, "echo \"Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\" >> bad_label.list",
                    truth.x, truth.y, truth.w, truth.h);
                system(buff);
            }
            int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
            if (class_id >= l.classes || class_id < 0) continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value

            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < l.total; ++n) {
                box pred = { 0 };
                pred.w = l.biases[2 * n] / state.net.w;
                pred.h = l.biases[2 * n + 1] / state.net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                if (l.map) class_id = l.map[class_id];

                int box_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                (*state.net.total_bbox)++;

                const int truth_in_index = t * l.truth_size + b * l.truths + 5;
                const int track_id = state.truth[truth_in_index];
                const int truth_out_index = b * l.n * l.w * l.h + mask_n * l.w * l.h + j * l.w + i;
                l.labels[truth_out_index] = track_id;
                l.class_ids[truth_out_index] = class_id;
                //printf(" track_id = %d, t = %d, b = %d, truth_in_index = %d, truth_out_index = %d \n", track_id, t, b, truth_in_index, truth_out_index);

                // range is 0 <= 1
                args->tot_iou += all_ious.iou;
                args->tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                args->tot_giou_loss += 1 - all_ious.giou;

                tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;

                tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;

                int obj_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
                avg_obj += l.output[obj_index];
                if (l.objectness_smooth) {
                    float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
                    if (l.delta[obj_index] == 0) l.delta[obj_index] = delta_obj;
                }
                else l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);

                int class_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

                //printf(" label: class_id = %d, truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", class_id, truth.x, truth.y, truth.w, truth.h);
                //printf(" mask_n = %d, l.output[obj_index] = %f, l.output[class_index + class_id] = %f \n\n", mask_n, l.output[obj_index], l.output[class_index + class_id]);

                ++(args->count);
                ++(args->class_count);
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;
            }

            // iou_thresh
            for (n = 0; n < l.total; ++n) {
                int mask_n = int_index(l.mask, n, l.n);
                if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) {
                    box pred = { 0 };
                    pred.w = l.biases[2 * n] / state.net.w;
                    pred.h = l.biases[2 * n + 1] / state.net.h;
                    float iou = box_iou_kind(pred, truth_shift, l.iou_thresh_kind); // IOU, GIOU, MSE, DIOU, CIOU
                    // iou, n

                    if (iou > l.iou_thresh) {
                        int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
                        if (l.map) class_id = l.map[class_id];

                        int box_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        ious all_ious = delta_yolo_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                        (*state.net.total_bbox)++;

                        // range is 0 <= 1
                        args->tot_iou += all_ious.iou;
                        args->tot_iou_loss += 1 - all_ious.iou;
                        // range is -1 <= giou <= 1
                        tot_giou += all_ious.giou;
                        args->tot_giou_loss += 1 - all_ious.giou;

                        tot_diou += all_ious.diou;
                        tot_diou_loss += 1 - all_ious.diou;

                        tot_ciou += all_ious.ciou;
                        tot_ciou_loss += 1 - all_ious.ciou;

                        int obj_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
                        avg_obj += l.output[obj_index];
                        if (l.objectness_smooth) {
                            float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
                            if (l.delta[obj_index] == 0) l.delta[obj_index] = delta_obj;
                        }
                        else l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);

                        int class_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

                        ++(args->count);
                        ++(args->class_count);
                        if (all_ious.iou > .5) recall += 1;
                        if (all_ious.iou > .75) recall75 += 1;
                    }
                }
            }
        }

        if (l.iou_thresh < 1.0f) {
            // averages the deltas obtained by the function: delta_yolo_box()_accumulate
            for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w; ++i) {
                    for (n = 0; n < l.n; ++n) {
                        int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                        int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        const int stride = l.w*l.h;

                        if (l.delta[obj_index] != 0)
                            averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta);
                    }
                }
            }
        }

    }

    return 0;
}



void forward_yolo_layer(const layer l, network_state state)
{
    //int i, j, b, t, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
    int b, n;

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int bbox_index = entry_index(l, b, n*l.w*l.h, 0);
            if (l.new_coords) {
                //activate_array(l.output + bbox_index, 4 * l.w*l.h, LOGISTIC);    // x,y,w,h
            }
            else {
                activate_array(l.output + bbox_index, 2 * l.w*l.h, LOGISTIC);        // x,y,
                int obj_index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array(l.output + obj_index, (1 + l.classes)*l.w*l.h, LOGISTIC);
            }
            scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + bbox_index, 1);    // scale x,y
        }
    }
#endif

    // delta is zeroed
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!state.train) return;

    int i;
    for (i = 0; i < l.batch * l.w*l.h*l.n; ++i) l.labels[i] = -1;
    for (i = 0; i < l.batch * l.w*l.h*l.n; ++i) l.class_ids[i] = -1;
    //float avg_iou = 0;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;


    int num_threads = l.batch;
    pthread_t* threads = (pthread_t*)calloc(num_threads, sizeof(pthread_t));

    struct train_yolo_args* yolo_args = (train_yolo_args*)xcalloc(l.batch, sizeof(struct train_yolo_args));

    for (b = 0; b < l.batch; b++)
    {
        yolo_args[b].l = l;
        yolo_args[b].state = state;
        yolo_args[b].b = b;

        yolo_args[b].tot_iou = 0;
        yolo_args[b].tot_iou_loss = 0;
        yolo_args[b].tot_giou_loss = 0;
        yolo_args[b].count = 0;
        yolo_args[b].class_count = 0;

        if (pthread_create(&threads[b], 0, process_batch, &(yolo_args[b]))) error("Thread creation failed", DARKNET_LOC);
    }

    for (b = 0; b < l.batch; b++)
    {
        pthread_join(threads[b], 0);

        tot_iou += yolo_args[b].tot_iou;
        tot_iou_loss += yolo_args[b].tot_iou_loss;
        tot_giou_loss += yolo_args[b].tot_giou_loss;
        count += yolo_args[b].count;
        class_count += yolo_args[b].class_count;
    }

    free(yolo_args);
    free(threads);

    // Search for an equidistant point from the distant boundaries of the local minimum
    int iteration_num = get_current_iteration(state.net);
    const int start_point = state.net.max_batches * 3 / 4;
    //printf(" equidistant_point ep = %d, it = %d \n", state.net.equidistant_point, iteration_num);

    if ((state.net.badlabels_rejection_percentage && start_point < iteration_num) ||
        (state.net.num_sigmas_reject_badlabels && start_point < iteration_num) ||
        (state.net.equidistant_point && state.net.equidistant_point < iteration_num))
    {
        const float progress_it = iteration_num - state.net.equidistant_point;
        const float progress = progress_it / (state.net.max_batches - state.net.equidistant_point);
        float ep_loss_threshold = (*state.net.delta_rolling_avg) * progress * 1.4;

        float cur_max = 0;
        float cur_avg = 0;
        float counter = 0;
        for (i = 0; i < l.batch * l.outputs; ++i) {

            if (l.delta[i] != 0) {
                counter++;
                cur_avg += fabs(l.delta[i]);

                if (cur_max < fabs(l.delta[i]))
                    cur_max = fabs(l.delta[i]);
            }
        }

        cur_avg = cur_avg / counter;

        if (*state.net.delta_rolling_max == 0) *state.net.delta_rolling_max = cur_max;
        *state.net.delta_rolling_max = *state.net.delta_rolling_max * 0.99 + cur_max * 0.01;
        *state.net.delta_rolling_avg = *state.net.delta_rolling_avg * 0.99 + cur_avg * 0.01;

        // reject high loss to filter bad labels
        if (state.net.num_sigmas_reject_badlabels && start_point < iteration_num)
        {
            const float rolling_std = (*state.net.delta_rolling_std);
            const float rolling_max = (*state.net.delta_rolling_max);
            const float rolling_avg = (*state.net.delta_rolling_avg);
            const float progress_badlabels = (float)(iteration_num - start_point) / (start_point);

            float cur_std = 0;
            float counter = 0;
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (l.delta[i] != 0) {
                    counter++;
                    cur_std += pow(l.delta[i] - rolling_avg, 2);
                }
            }
            cur_std = sqrt(cur_std / counter);

            *state.net.delta_rolling_std = *state.net.delta_rolling_std * 0.99 + cur_std * 0.01;

            float final_badlebels_threshold = rolling_avg + rolling_std * state.net.num_sigmas_reject_badlabels;
            float badlabels_threshold = rolling_max - progress_badlabels * fabs(rolling_max - final_badlebels_threshold);
            badlabels_threshold = max_val_cmp(final_badlebels_threshold, badlabels_threshold);
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (fabs(l.delta[i]) > badlabels_threshold)
                    l.delta[i] = 0;
            }
            printf(" rolling_std = %f, rolling_max = %f, rolling_avg = %f \n", rolling_std, rolling_max, rolling_avg);
            printf(" badlabels loss_threshold = %f, start_it = %d, progress = %f \n", badlabels_threshold, start_point, progress_badlabels *100);

            ep_loss_threshold = min_val_cmp(final_badlebels_threshold, rolling_avg) * progress;
        }


        // reject some percent of the highest deltas to filter bad labels
        if (state.net.badlabels_rejection_percentage && start_point < iteration_num) {
            if (*state.net.badlabels_reject_threshold == 0)
                *state.net.badlabels_reject_threshold = *state.net.delta_rolling_max;

            printf(" badlabels_reject_threshold = %f \n", *state.net.badlabels_reject_threshold);

            const float num_deltas_per_anchor = (l.classes + 4 + 1);
            float counter_reject = 0;
            float counter_all = 0;
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (l.delta[i] != 0) {
                    counter_all++;
                    if (fabs(l.delta[i]) > (*state.net.badlabels_reject_threshold)) {
                        counter_reject++;
                        l.delta[i] = 0;
                    }
                }
            }
            float cur_percent = 100 * (counter_reject*num_deltas_per_anchor / counter_all);
            if (cur_percent > state.net.badlabels_rejection_percentage) {
                *state.net.badlabels_reject_threshold += 0.01;
                printf(" increase!!! \n");
            }
            else if (*state.net.badlabels_reject_threshold > 0.01) {
                *state.net.badlabels_reject_threshold -= 0.01;
                printf(" decrease!!! \n");
            }

            printf(" badlabels_reject_threshold = %f, cur_percent = %f, badlabels_rejection_percentage = %f, delta_rolling_max = %f \n",
                *state.net.badlabels_reject_threshold, cur_percent, state.net.badlabels_rejection_percentage, *state.net.delta_rolling_max);
        }


        // reject low loss to find equidistant point
        if (state.net.equidistant_point && state.net.equidistant_point < iteration_num) {
            printf(" equidistant_point loss_threshold = %f, start_it = %d, progress = %3.1f %% \n", ep_loss_threshold, state.net.equidistant_point, progress * 100);
            for (i = 0; i < l.batch * l.outputs; ++i) {
                if (fabs(l.delta[i]) < ep_loss_threshold)
                    l.delta[i] = 0;
            }
        }
    }

    if (count == 0) count = 1;
    if (class_count == 0) class_count = 1;

    if (l.show_details == 0) {
        float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
        *(l.cost) = loss;

        loss /= l.batch;

        fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f), count: %d, total_loss = %f \n",
            (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, state.index, tot_iou / count, count, loss);
    }
    else {
        // show detailed output

        int stride = l.w*l.h;
        float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
        memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));


        int j, n;
        for (b = 0; b < l.batch; ++b) {
            for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w; ++i) {
                    for (n = 0; n < l.n; ++n) {
                        int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                        no_iou_loss_delta[index + 0 * stride] = 0;
                        no_iou_loss_delta[index + 1 * stride] = 0;
                        no_iou_loss_delta[index + 2 * stride] = 0;
                        no_iou_loss_delta[index + 3 * stride] = 0;
                    }
                }
            }
        }

        float classification_loss = l.obj_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
        free(no_iou_loss_delta);
        float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
        float iou_loss = loss - classification_loss;

        float avg_iou_loss = 0;
        *(l.cost) = loss;

        // gIOU loss + MSE (objectness) loss
        if (l.iou_loss == MSE) {
            *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
        }
        else {
            // Always compute classification loss both for iou + cls loss and for logging with mse loss
            // TODO: remove IOU loss fields before computing MSE on class
            //   probably split into two arrays
            if (l.iou_loss == GIOU) {
                avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
            }
            else {
                avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
            }
            *(l.cost) = avg_iou_loss + classification_loss;
        }


        loss /= l.batch;
        classification_loss /= l.batch;
        iou_loss /= l.batch;

        fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f), count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
            (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, state.index, tot_iou / count, count, classification_loss, iou_loss, loss);

        //fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
        //    (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
        //    classification_loss, iou_loss, loss);
    }
}

void backward_yolo_layer(const layer l, network_state state)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

// Converts output of the network to detection boxes
// w,h: image width,height
// netw,neth: network width,height
// relative: 1 (all callers seems to pass TRUE)
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    // network height (or width)
    int new_w = 0;
    // network height (or width)
    int new_h = 0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
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
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w / netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h / neth;
    for (i = 0; i < n; ++i) {

        box b = dets[i].bbox;
        // x = ( x - (deltaw/2)/netw ) / ratiow;
        //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
        b.x = (b.x - deltaw / 2. / netw) / ratiow;
        b.y = (b.y - deltah / 2. / neth) / ratioh;
        // scale to match rotation of incoming image
        b.w *= 1 / ratiow;
        b.h *= 1 / ratioh;

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }

        dets[i].bbox = b;
    }
}

/*
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
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
*/

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for(n = 0; n < l.n; ++n){
        for (i = 0; i < l.w*l.h; ++i) {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

int yolo_num_detections_batch(layer l, float thresh, int batch)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, batch, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
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

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    // This snippet below is not necessary
    // Need to comment it in order to batch processing >= 2 images
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.new_coords);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                if (l.embedding_output) {
                    get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, 0, dets[count].embeddings);
                }

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

int get_yolo_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch)
{
    int i,j,n;
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, batch, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, batch, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.new_coords);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                if (l.embedding_output) {
                    get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, batch, dets[count].embeddings);
                }

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, batch, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    if (l.embedding_output) {
        layer le = state.net.layers[l.embedding_layer_id];
        cuda_pull_array_async(le.output_gpu, l.embedding_output, le.batch*le.outputs);
    }

    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int bbox_index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            if (l.new_coords) {
                //activate_array_ongpu(l.output_gpu + bbox_index, 4 * l.w*l.h, LOGISTIC);    // x,y,w,h
            }
            else {
                activate_array_ongpu(l.output_gpu + bbox_index, 2 * l.w*l.h, LOGISTIC);    // x,y

                int obj_index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array_ongpu(l.output_gpu + obj_index, (1 + l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
            }
            if (l.scale_x_y != 1) scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + bbox_index, 1);      // scale x,y
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        if (l.mean_alpha && l.output_avg_gpu) mean_array_gpu(l.output_gpu, l.batch*l.outputs, l.mean_alpha, l.output_avg_gpu);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, state.net.loss_scale * l.delta_normalizer, l.delta_gpu, 1, state.delta, 1);
}
#endif
