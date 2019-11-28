#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

/*
IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}
*/

Mat image_to_mat(image im)
{
    cv::Mat m(im.h, im.w, CV_8UC(im.c));
    int step = m.step;
    for(int y = 0; y < im.h; ++y){
        for(int x = 0; x < im.w; ++x){
            for(int c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                m.data[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return m;
}

image mat_to_image(Mat m)
{
    int h = m.rows;
    int w = m.cols;
    int c = m.channels();
    image im = make_image(w, h, c);
    return *mat_to_image(m, &im);
}

image* mat_to_image(Mat m, image* im)
{
    // m.type() assumed to be CV8UCX, 0 <= X < 3
    int h = m.rows;
    int w = m.cols;
    int c = m.channels();
    unsigned char *data = (unsigned char *)m.data;
    int step = m.step;

    for(int i = 0; i < h; ++i){
        for(int k= 0; k < c; ++k){
            for(int j = 0; j < w; ++j){
                im->data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(cv::CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(cv::CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(cv::CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}
#endif
