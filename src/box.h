#ifndef BOX_H
#define BOX_H

#include "darknet.h"

//typedef struct{
//    float x, y, w, h;
//} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

//typedef struct detection {
//	box bbox;
//	int classes;
//	float *prob;
//	float *mask;
//	float objectness;
//	int sort_class;
//} detection;

typedef struct detection_with_class {
	detection det;
	// The most probable class id: the best class index in this->prob.
	// Is filled temporary when processing results, otherwise not initialized
	int best_class;
} detection_with_class;

#ifdef __cplusplus
extern "C" {
#endif
box float_to_box(float *f);
box float_to_box_stride(float *f, int stride);
float box_iou(box a, box b);
float box_iou_kind(box a, box b, IOU_LOSS iou_kind);
float box_rmse(box a, box b);
dxrep dx_box_iou(box a, box b, IOU_LOSS iou_loss);
float box_giou(box a, box b);
float box_diou(box a, box b);
float box_ciou(box a, box b);
dbox diou(box a, box b);
boxabs to_tblr(box a);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh);
//LIB_API void do_nms_sort(detection *dets, int total, int classes, float thresh);
//LIB_API void do_nms_obj(detection *dets, int total, int classes, float thresh);
//LIB_API void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

// Creates array of detections with prob > thresh and fills best_class for them
// Return number of selected detections in *selected_detections_num
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num, char **names);

#ifdef __cplusplus
}
#endif
#endif
