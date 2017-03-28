#ifndef BOX_H
#define BOX_H

	typedef struct{
		float x, y, w, h;
	} box;

	typedef struct{
		float dx, dy, dw, dh;
	} dbox;

#ifdef __cplusplus
	extern "C" {
#endif
box float_to_box(float *f, int stride);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#ifdef __cplusplus
}
#endif


#endif
