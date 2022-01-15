#ifdef OPENCV
#ifdef __cplusplus

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifdef OPENCV
    #include <opencv2/opencv.hpp>
    #include <opencv2/highgui.hpp>
    #include <opencv2/highgui/highgui_c.h>
#endif

#include "image.h"

using namespace cv;

extern "C" {


image cv_copy_image(image p) {
    image copy;
    copy.data = (float *) calloc(p.h * p.w * p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));
    copy.h = p.h;
    copy.w = p.w;
    copy.c = p.c;
    return copy;
}

image copy_image_cv(image p)
{
    image copy = p;
    copy.data = (float *)calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void *image_to_ipl_cv(image im) {
    int x, y, c;
    IplImage *disp = cvCreateImage(cvSize(im.w, im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for (y = 0; y < im.h; ++y) {
        for (x = 0; x < im.w; ++x) {
            for (c = 0; c < im.c; ++c) {
                float val = im.data[c * im.h * im.w + y * im.w + x];
                disp->imageData[y * step + x * im.c + c] = (unsigned char) (val * 255);
            }
        }
    }
    return disp;
}

image make_empty_image_cv(int w, int h, int c) {
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image_cv(int w, int h, int c) {
    image out = make_empty_image_cv(w, h, c);
    out.data = (float *) calloc(h * w * c, sizeof(float));
    return out;
}

void ipl_into_image_cv(void *s, image im) {
    IplImage *src = (IplImage *)s;
    unsigned char *data = (unsigned char *) src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for (i = 0; i < h; ++i) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < w; ++j) {
                im.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 255.;
            }
        }
    }
}

image ipl_to_image_cv(void *s) {
    IplImage *src = (IplImage *)s;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image_cv(w, h, c);
    ipl_into_image_cv(src, out);
    return out;
}

void free_image_cv(image m) {
    if (m.data) {
        free(m.data);
    }
}

void rgbgr_image_cv(image im) {
    int i;
    for (i = 0; i < im.w * im.h; ++i) {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w * im.h * 2];
        im.data[i + im.w * im.h * 2] = swap;
    }
}

void constrain_image_cv(image im) {
    int i;
    for (i = 0; i < im.w * im.h * im.c; ++i) {
        if (im.data[i] < 0) im.data[i] = 0;
        if (im.data[i] > 1) im.data[i] = 1;
    }
}

Mat image_to_mat_cv(image img) {
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    Mat mat = Mat(height, width, CV_8UC(channels));
    int step = mat.step;
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c * img.h * img.w + y * img.w + x];
                mat.data[y * step + x * img.c + c] = (unsigned char) (val * 255);
            }
        }
    }
    return mat;
}

image mat_to_image_cv(Mat mat) {
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image_cv(w, h, c);
    unsigned char *data = (unsigned char *) mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                im.data[k * w * h + y * w + x] = data[y * step + x * c + k] / 255.0f;
            }
        }
    }
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps) {
    VideoCapture *cap;
    if (f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if (!cap->isOpened()) return cap;
    if (w) cap->set(CAP_PROP_FRAME_WIDTH, w);
    if (h) cap->set(CAP_PROP_FRAME_HEIGHT, h);
    if (fps) cap->set(CAP_PROP_FPS, fps);
    return (void *) cap;
}

image get_image_from_stream_cv(void *p) {
    VideoCapture *cap = (VideoCapture *)p;
    Mat frame;
    (*cap).read(frame);
    return mat_to_image_cv(frame);
}

int show_image_frame_cv(image im, const char *name, int ms) {
    Mat m = image_to_mat_cv(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c % 256;
    m.release();
    return c;
}

void make_window_cv(char *name, int w, int h, int fullscreen) {
    namedWindow(name, CV_WINDOW_NORMAL);
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if (strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

int cv_wait_key(int k) {
    return cvWaitKey(k);
}

void *cv_capture_from_file(const char *filename) {
    VideoCapture *cap = new VideoCapture(filename);
    return cap;
}

void *cv_create_image(image *buff) {
    return cvCreateImage(cvSize(buff[0].w, buff[0].h), IPL_DEPTH_8U, buff[0].c);
}

void *cv_capture_from_camera(int cam_index, int w, int h, int frames) {
    VideoCapture *cap = new VideoCapture(cam_index);

    if (w) {
        cap->set(CAP_PROP_FRAME_WIDTH, w);
    }
    if (h) {
        cap->set(CAP_PROP_FRAME_HEIGHT, h);
    }
    if (frames) {
        cap->set(CAP_PROP_FPS, frames);
    }

    return cap;
}

void cv_create_named_window(int fullscreen, int w, int h) {
    cvNamedWindow("Demo", CV_WINDOW_NORMAL);
    if (fullscreen) {
        cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        cvMoveWindow("Demo", 0, 0);
        cvResizeWindow("Demo", w, h);
    }
}

static float get_pixel(image m, int x, int y, int c) {
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}

void show_image_cv(image p, const char *name) {
    int x, y, k;
    //if (p.c == 3) rgbgr_image_cv(p);

    char buff[1024];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    IplImage *img = (IplImage *)image_to_ipl_cv(p);

    cvShowImage(buff, img);

    cvReleaseImage(&img);
}

int cv_show_image(image p, const char *name, int ms) {
    image copy = copy_image_cv(p);
    constrain_image_cv(copy);
    //if (copy.c == 3) rgbgr_image_cv(copy);
    show_image_cv(copy, name);
    int c = cvWaitKey(ms);
    if (c != -1) c = c % 256;
    free_image_cv(copy);
    return c;
}

void image_into_ipl(image im, void *d) {
    IplImage *dst = (IplImage *)d;
    int x, y, k;
    //if (im.c == 3) rgbgr_image_cv(im);

    int step = dst->widthStep;

    for (y = 0; y < im.h; ++y) {
        for (x = 0; x < im.w; ++x) {
            for (k = 0; k < im.c; ++k) {
                dst->imageData[y * step + x * im.c + k] = (unsigned char) (get_pixel(im, x, y, k) * 255);
            }
        }
    }
}

void ipl_into_image(IplImage *src, image im) {
    unsigned char *data = (unsigned char *) src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for (i = 0; i < h; ++i) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < w; ++j) {
                im.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 255.;
            }
        }
    }
}

image ipl_to_image(void *src) {
    IplImage *s = (IplImage *) src;
    int h = s->height;
    int w = s->width;
    int c = s->nChannels;
    image out = make_image_cv(w, h, c);
    ipl_into_image(s, out);
    return out;
}

int is_invalid_file_to_imread_cv(char *filename, int channels) {
    if (filename) filename[strcspn(filename, "\n\r")] = 0;

    Mat src;

    try {
        src = imread(filename, channels == 1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
    }
    catch(...) {

    }

    if (!src.data) {
        return 1;
    }

    src.release();

    return 0;
}

image load_image_cv(char *filename, int channels) {
    if (filename) filename[strcspn(filename, "\n\r")] = 0;

    char *pos;
    if ((pos = strchr(filename, '\r')) != NULL) *pos = '\0';
    if ((pos = strchr(filename, '\n')) != NULL) *pos = '\0';

    Mat src;

    try {
        src = imread(filename, channels == 1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
    }
    catch(...) { }

    if (src.empty()) {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);

        char buff[1024];

        // Check the length of the buffer
        if (strlen(filename) > 1024) {
            sprintf(buff, "This filename is too long");
        } else {
            sprintf(buff, "%s", filename);
        }

        // Write directly to the file rather than using the system call to write
        FILE *bad_list = fopen("bad.list", "a");
        fwrite(buff, sizeof(char), strlen(buff), bad_list);
        fwrite("\n", 1, 1, bad_list);

        //exit(0);
        return make_image_cv(10, 10, 3);
    }

    image out = mat_to_image_cv(src);

    src.release();

    //rgbgr_image_cv(out);

    return out;
}

void save_image_jpg_cv(image p, const char *name) {
    image copy = copy_image_cv(p);
    //if (p.c == 3) rgbgr_image_cv(copy);
    int x, y, k;

    char buff[1024];
    sprintf(buff, "%s.jpg", name);

    IplImage *disp = (IplImage *)cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    for (y = 0; y < p.h; ++y) {
        for (x = 0; x < p.w; ++x) {
            for (k = 0; k < p.c; ++k) {
                disp->imageData[y * step + x * p.c + k] = (unsigned char) (get_pixel(copy, x, y, k) * 255);
            }
        }
    }

    imwrite(buff, cvarrToMat(disp, false));

    cvReleaseImage(&disp);
    free_image_cv(copy);
}

static void cv_set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

int cv_constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

image cv_crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image_cv(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = cv_constrain_int(r, 0, im.h-1);
                c = cv_constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                cv_set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

} // extern "C"
#endif
#endif
