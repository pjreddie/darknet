#ifndef BOX_H
#define BOX_H

typedef struct{
    float x, y, w, h;
} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_iou(box a, box b);
dbox diou(box a, box b);

#endif
