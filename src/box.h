#ifndef BOX_H
#define BOX_H
#include "darknet.h"

typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_rmse(dn_box a, dn_box b);
dbox diou(dn_box a, dn_box b);
dn_box decode_box(dn_box b, dn_box anchor);
dn_box encode_box(dn_box b, dn_box anchor);

#endif
