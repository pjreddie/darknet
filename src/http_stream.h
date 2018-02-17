#pragma once
#ifndef HTTP_STREAM_H
#define HTTP_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

void send_mjpeg(IplImage* ipl, int port, int timeout, int quality);

#ifdef __cplusplus
}
#endif

#endif // HTTP_STREAM_H