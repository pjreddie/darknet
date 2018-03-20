#pragma once
#ifndef HTTP_STREAM_H
#define HTTP_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

void send_mjpeg(IplImage* ipl, int port, int timeout, int quality);
CvCapture* get_capture_webcam(int index);
IplImage* get_webcam_frame(CvCapture *cap);

#ifdef __cplusplus
}
#endif

#endif // HTTP_STREAM_H