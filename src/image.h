#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "darknet.h"

#ifdef OPENCV
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *x;
    float *y;
} float_pair;

image ipl_to_image(void *src);
image load_image_cv(char *filename, int channels);
image get_image_from_stream_cv(void *cap);
void blur_image_and_save_cv(image im, int num, int classes, detection *dets, float thresh, const char *fname);
void save_image_jpg_cv(image p, const char *name);
void *open_video_stream(const char *f, int c, int w, int h, int fps);
int cv_wait_key(int key);
void *cv_capture_from_file(const char* filename);
void *cv_capture_from_camera(int cam_index, int w, int h, int frames);
void *cv_create_image(image* buff);
void cv_create_named_window(int fullscreen, int w, int h);
int cv_show_image(image p, const char *name, int ms);
void save_image_jpg_cv(image p, const char *name);
void extract_voxel_cv(char *lfile, char *rfile, char *prefix, int w, int h);

#ifdef __cplusplus
}
#endif
#endif

float get_color(int c, int x, int max);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_bbox(image a, box bbox, int w, float r, float g, float b);
void write_label(image a, int r, int c, image *characters, char *string, float *rgb);
image image_distance(image a, image b);
void scale_image(image m, float s);
image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect);
image random_crop_image(image im, int w, int h);
image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h);
augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h);
void letterbox_image_into(image im, int w, int h, image boxed);
image resize_max(image im, int max);
void translate_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
void place_image(image im, int w, int h, int dx, int dy, image canvas);
void saturate_image(image im, float sat);
void exposure_image(image im, float sat);
void distort_image(image im, float hue, float sat, float val);
void saturate_exposure_image(image im, float sat, float exposure);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void yuv_to_rgb(image im);
void rgb_to_yuv(image im);

image collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
image collapse_images_vert(image *ims, int n);

void show_image_normalized(image im, const char *name);
void show_images(image *ims, int n, char *window);
void show_image_layers(image p, char *name);
void show_image_collapsed(image p, char *name);

void print_image(image m);

image make_empty_image(int w, int h, int c);
void copy_image_into(image src, image dest);

image get_image_layer(image m, int l);

void reconstruct_picture(network net, float *features, image recon, image update, float rate, float momentum, float lambda, int smooth_size, int iters);

#endif
