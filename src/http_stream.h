#pragma once
#ifndef HTTP_STREAM_H
#define HTTP_STREAM_H

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "image.h"
#include <stdint.h>

#ifdef OPENCV
void send_mjpeg(IplImage* ipl, int port, int timeout, int quality);
CvCapture* get_capture_webcam(int index);
CvCapture* get_capture_video_stream(char *path);
IplImage* get_webcam_frame(CvCapture *cap);
int get_stream_fps_cpp(CvCapture *cap);

image image_data_augmentation(IplImage* ipl, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float jitter, float dhue, float dsat, float dexp);
#endif  // OPENCV

double get_time_point();
void start_timer();
void stop_timer();
double get_time();
void stop_timer_and_show();
void stop_timer_and_show_name(char *name);
void show_total_time();

#ifdef __cplusplus
}
#endif

#endif // HTTP_STREAM_H