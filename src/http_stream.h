#pragma once
#ifndef HTTP_STREAM_H
#define HTTP_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif
#include "image.h"

void send_mjpeg(IplImage* ipl, int port, int timeout, int quality);
CvCapture* get_capture_webcam(int index);
CvCapture* get_capture_video_stream(char *path);
IplImage* get_webcam_frame(CvCapture *cap);

image image_data_augmentation(IplImage* ipl, int w, int h,
	int pleft, int ptop, int swidth, int sheight, int flip,
	float jitter, float dhue, float dsat, float dexp);

#ifdef __cplusplus
}
#endif

#endif // HTTP_STREAM_H