#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

box float_to_box(float *f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}

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

// where c is the smallest box that fully encompases a and b
boxabs box_c(box a, box b) {
    boxabs ba = { 0 };
    ba.top = fmin(a.y - a.h / 2, b.y - b.h / 2);
    ba.bot = fmax(a.y + a.h / 2, b.y + b.h / 2);
    ba.left = fmin(a.x - a.w / 2, b.x - b.w / 2);
    ba.right = fmax(a.x + a.w / 2, b.x + b.w / 2);
    return ba;
}

// representation from x, y, w, h to top, left, bottom, right
boxabs to_tblr(box a) {
    boxabs tblr = { 0 };
    float t = a.y - (a.h / 2);
    float b = a.y + (a.h / 2);
    float l = a.x - (a.w / 2);
    float r = a.x + (a.w / 2);
    tblr.top = t;
    tblr.bot = b;
    tblr.left = l;
    tblr.right = r;
    return tblr;
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
    //return box_intersection(a, b)/box_union(a, b);

    float I = box_intersection(a, b);
    float U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}

float box_giou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w*h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float u = box_union(a, b);
    float giou_term = (c - u) / c;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, giou_term: %f\n", c, u, giou_term);
#endif
    return iou - giou_term;
}

dxrep dx_box_iou(box pred, box truth, IOU_LOSS iou_loss) {
    boxabs pred_tblr = to_tblr(pred);
    float pred_t = fmin(pred_tblr.top, pred_tblr.bot);
    float pred_b = fmax(pred_tblr.top, pred_tblr.bot);
    float pred_l = fmin(pred_tblr.left, pred_tblr.right);
    float pred_r = fmax(pred_tblr.left, pred_tblr.right);

    boxabs truth_tblr = to_tblr(truth);
#ifdef DEBUG_PRINTS
    printf("\niou: %f, giou: %f\n", box_iou(pred, truth), box_giou(pred, truth));
    printf("pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
    printf("truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
#endif
    //printf("pred (t,b,l,r): (%f, %f, %f, %f)\n", pred_t, pred_b, pred_l, pred_r);
    //printf("trut (t,b,l,r): (%f, %f, %f, %f)\n", truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
    dxrep dx = { 0 };
    float X = (pred_b - pred_t) * (pred_r - pred_l);
    float Xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
    float Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
    float Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
    float I = Iw * Ih;
    float U = X + Xhat - I;

    float Cw = fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
    float Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
    float C = Cw * Ch;

    // float IoU = I / U;
    // Partial Derivatives, derivatives
    float dX_wrt_t = -1 * (pred_r - pred_l);
    float dX_wrt_b = pred_r - pred_l;
    float dX_wrt_l = -1 * (pred_b - pred_t);
    float dX_wrt_r = pred_b - pred_t;

    // gradient of I min/max in IoU calc (prediction)
    float dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
    float dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
    float dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
    float dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
    // derivative of U with regard to x
    float dU_wrt_t = dX_wrt_t - dI_wrt_t;
    float dU_wrt_b = dX_wrt_b - dI_wrt_b;
    float dU_wrt_l = dX_wrt_l - dI_wrt_l;
    float dU_wrt_r = dX_wrt_r - dI_wrt_r;
    // gradient of C min/max in IoU calc (prediction)
    float dC_wrt_t = pred_t < truth_tblr.top ? (-1 * Cw) : 0;
    float dC_wrt_b = pred_b > truth_tblr.bot ? Cw : 0;
    float dC_wrt_l = pred_l < truth_tblr.left ? (-1 * Ch) : 0;
    float dC_wrt_r = pred_r > truth_tblr.right ? Ch : 0;

    // Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
    float p_dt = 0;
    float p_db = 0;
    float p_dl = 0;
    float p_dr = 0;
    if (U > 0) {
        p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
        p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
        p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
        p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
    }

    if (iou_loss == GIOU) {
        if (C > 0) {
            // apply "C" term from gIOU
            p_dt += ((C * dU_wrt_t) - (U * dC_wrt_t)) / (C * C);
            p_db += ((C * dU_wrt_b) - (U * dC_wrt_b)) / (C * C);
            p_dl += ((C * dU_wrt_l) - (U * dC_wrt_l)) / (C * C);
            p_dr += ((C * dU_wrt_r) - (U * dC_wrt_r)) / (C * C);
        }
    }

    // apply grad from prediction min/max for correct corner selection
    dx.dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    dx.db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    dx.dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    dx.dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    return dx;
}

float box_rmse(box a, box b)
{
    return sqrt(pow(a.x-b.x, 2) +
                pow(a.y-b.y, 2) +
                pow(a.w-b.w, 2) +
                pow(a.h-b.h, 2));
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

typedef struct{
    int index;
    int class_id;
    float **probs;
} sortable_bbox;

int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class_id] - b.probs[b.index][b.class_id];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox* s = (sortable_bbox*)calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;
        s[i].class_id = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class_id = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

int nms_comparator_v3(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (i = 0; i < total; ++i) {
        dets[i].sort_class = -1;
    }

    qsort(dets, total, sizeof(detection), nms_comparator_v3);
    for (i = 0; i < total; ++i) {
        if (dets[i].objectness == 0) continue;
        box a = dets[i].bbox;
        for (j = i + 1; j < total; ++j) {
            if (dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
            if (box_iou(a, b) > thresh) {
                dets[j].objectness = 0;
                for (k = 0; k < classes; ++k) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i) {
            //printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}

box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}

box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}
