#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
image load_image_cv(const char *filename, int channels);
int show_image_cv(image im, const char* name, int ms);
#endif

float get_color(int c, int x, int max);
void draw_box(dn_image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_bbox(dn_image a, dn_box bbox, int w, float r, float g, float b);
void write_label(dn_image a, int r, int c, dn_image *characters, char *string, float *rgb);
dn_image image_distance(dn_image a, dn_image b);
void scale_image(dn_image m, float s);
dn_image rotate_crop_image(dn_image im, float rad, float s, int w, int h, float dx, float dy, float aspect);
dn_image random_crop_image(dn_image im, int w, int h);
dn_image random_augment_image(dn_image im, float angle, float aspect, int low, int high, int w, int h);
augment_args random_augment_args(dn_image im, float angle, float aspect, int low, int high, int w, int h);
void letterbox_image_into(dn_image im, int w, int h, dn_image boxed);
dn_image resize_max(dn_image im, int max);
void translate_image(dn_image m, float s);
void embed_image(dn_image source, dn_image dest, int dx, int dy);
void place_image(dn_image im, int w, int h, int dx, int dy, dn_image canvas);
void saturate_image(dn_image im, float sat);
void exposure_image(dn_image im, float sat);
void distort_image(dn_image im, float hue, float sat, float val);
void saturate_exposure_image(dn_image im, float sat, float exposure);
void rgb_to_hsv(dn_image im);
void hsv_to_rgb(dn_image im);
void yuv_to_rgb(dn_image im);
void rgb_to_yuv(dn_image im);


dn_image collapse_image_layers(dn_image source, int border);
dn_image collapse_images_horz(dn_image *ims, int n);
dn_image collapse_images_vert(dn_image *ims, int n);

void show_image_normalized(dn_image im, const char *name);
void show_images(dn_image *ims, int n, char *window);
void show_image_layers(dn_image p, char *name);
void show_image_collapsed(dn_image p, char *name);

void print_image(dn_image m);

dn_image make_empty_image(int w, int h, int c);
void copy_image_into(dn_image src, dn_image dest);

dn_image get_image_layer(dn_image m, int l);

#ifdef __cplusplus
}
#endif

#endif

