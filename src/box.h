#ifndef BOX_H
#define BOX_H
#include "darknet.h"

	typedef struct{
		float dx, dy, dw, dh;
	} dbox;

#ifdef __cplusplus
	extern "C" {
#endif
float box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#ifdef __cplusplus
}
#endif


#endif
