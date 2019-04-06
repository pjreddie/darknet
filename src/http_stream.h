#ifndef HTTP_STREAM_H
#define HTTP_STREAM_H
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "image.h"
#include <stdint.h>

void send_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, int port, int timeout);

#ifdef OPENCV
void send_mjpeg(mat_cv* mat, int port, int timeout, int quality);

#endif  // OPENCV

#ifdef __cplusplus
}
#endif

#endif // HTTP_STREAM_H
