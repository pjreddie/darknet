#ifndef BOX_H
#define BOX_H
#include "darknet.h"

typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

typedef struct detection_with_class {
    detection det;
    int best_class;
} detection_with_class;

box float_to_box(float *f, int stride);

box float_to_box_v4(float *f);
box float_to_box_stride_y4(float *f, int stride);
float box_iou_y4(box a, box b);
float box_iou_kind_y4(box a, box b, IOU_LOSS iou_kind);
float box_rmse_y4(box a, box b);
dxrep dx_box_iou_y4(box a, box b, IOU_LOSS iou_loss);
float box_giou_y4(box a, box b);
float box_diou_y4(box a, box b);
float box_ciou_y4(box a, box b);
dbox diou_y4(box a, box b);
boxabs to_tblr_y4(box a);
void do_nms_y4(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh);
box decode_box_y4(box b, box anchor);
box encode_box_y4(box b, box anchor);

detection_with_class* get_actual_detections_y4(detection *dets, int dets_num, float thresh, int* selected_detections_num, char **names);

#endif