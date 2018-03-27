#ifndef BOX_H
#define BOX_H

typedef struct{
    float x, y, w, h;
} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

typedef struct detection {
	box bbox;
	int classes;
	float *prob;
	float *mask;
	float objectness;
	int sort_class;
} detection;

box float_to_box(float *f);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v3(detection *dets, int total, int classes, float thresh);
void do_nms_obj_v3(detection *dets, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
