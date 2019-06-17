 #ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"
#include "darknet.h"
using namespace cv;
static int checkblur = 1;
extern int cando;
extern "C" {

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
    image im = make_image(w, h, c); // ÃÊ±âÈ­
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep; // image width * channel
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

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
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
    Mat dst; // blur image
    image im;
    //int min;
    m = imread(filename, flag); // read image
    if(!m.data){ // can't load image
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        cando = 0;
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    else
    {
        cando = 1;
    }
    
    if(checkblur == 1)
    {
        GaussianBlur(m,dst,Size(7,7),0);// blur
        /*   
        if(m.size().width<m.size().height)
        {
            min = m.size().width;
        }
        else
        {
            min = m.size().height;
        }

        if(min < 50)
        {
            GaussianBlur(m,dst,Size(3,3),0);// blur
        }
        else if(min < 100)
        {
            GaussianBlur(m,dst,Size(5,5),0);// blur
        }
        else if(min <300)
        {
            GaussianBlur(m,dst,Size(7,7),0);// blur
        }
        else if(min <500)
        {
            GaussianBlur(m,dst,Size(9,9),0);// blur
        }
        else if(min < 1000)
        {
            GaussianBlur(m,dst,Size(11,11),0);// blur
        }
        else
        {
            GaussianBlur(m,dst,Size(15,15),0);// blur
        }
       */  
        im = mat_to_image(dst); // blur image send to function
        checkblur = 0;
    }
    else
    {
        //printf("None GaussianBlur\n");
        im = mat_to_image(m);
        checkblur = 1;
    }
  
    //im = mat_to_image(m);
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
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

}

#endif
