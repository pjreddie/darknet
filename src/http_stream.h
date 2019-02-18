#ifndef HTTP_STREAM_H
#define HTTP_STREAM_H
#include "darknet.h"

#ifdef OPENCV
#include <opencv2/core/version.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio_c.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "image.h"
#include <stdint.h>

#ifdef OPENCV
void send_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, int port, int timeout);
void send_mjpeg(IplImage* ipl, int port, int timeout, int quality);
CvCapture* get_capture_webcam(int index);
CvCapture* get_capture_video_stream(const char *path);
IplImage* get_webcam_frame(CvCapture *cap);
int get_stream_fps_cpp(CvCapture *cap);

image image_data_augmentation(IplImage* ipl, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float jitter, float dhue, float dsat, float dexp);

image load_image_resize(char *filename, int w, int h, int c, image *im);
#endif  // OPENCV

#ifdef __cplusplus
}
#endif

#endif // HTTP_STREAM_H
