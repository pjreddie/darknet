#include "image_opencv.h"

#ifdef OPENCV
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <opencv2/world.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

using std::cerr;
using std::endl;

// OpenCV libraries
#ifndef CV_VERSION_EPOCH
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#else   // CV_VERSION_EPOCH
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH

// OpenCV includes
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#endif

#include "http_stream.h"

#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar( (b), (g), (r), 0 )
#endif

extern "C" {

    struct mat_cv :IplImage { int a[0]; };
    struct cap_cv : cv::VideoCapture { int a[0]; };
    struct write_cv : cv::VideoWriter { int a[0]; };

// ====================================================================
// cv::Mat / IplImage
// ====================================================================
image ipl_to_image(IplImage* src);

image load_image_cv(char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = 1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if ((src = cvLoadImage(filename, flag)) == 0)
    {
        char shrinked_filename[1024];
        if (strlen(filename) >= 1024) sprintf(shrinked_filename, "name is too long");
        else sprintf(shrinked_filename, "%s", filename);
        fprintf(stderr, "Cannot load image \"%s\"\n", shrinked_filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(shrinked_filename, sizeof(char), strlen(shrinked_filename), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);
        //if (check_mistakes) getchar();
        return make_image(10, 10, 3);
        //exit(EXIT_FAILURE);
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    if (out.c > 1)
        rgbgr_image(out);
    return out;
}
// ----------------------------------------

mat_cv *load_image_mat_cv(const char *filename, int flag)
{
    return (mat_cv *)cvLoadImage(filename, flag);
}
// ----------------------------------------


image load_image_resize(char *filename, int w, int h, int c, image *im)
{
    image out;
    cv::Mat img(h, w, CV_8UC3);
    try {
        int flag = -1;
        if (c == 0) flag = 1;
        else if (c == 1) { flag = 0; img = cv::Mat(h, w, CV_8UC1); }
        else if (c == 3) { flag = 1; img = cv::Mat(h, w, CV_8UC3); }
        else {
            cerr << "OpenCV can't force load with " << c << " channels\n";
        }
        //throw std::runtime_error("runtime_error");
        cv::Mat loaded_image = cv::imread(filename, flag);
        cv::cvtColor(loaded_image, loaded_image, cv::COLOR_RGB2BGR);
        IplImage tmp1 = loaded_image;
        *im = ipl_to_image(&tmp1);

        cv::resize(loaded_image, img, cv::Size(w, h), 0, 0, CV_INTER_LINEAR);

        IplImage tmp2 = img;
        out = ipl_to_image(&tmp2);
    }
    catch (...) {
        cerr << "OpenCV can't load image %s " << filename << " \n";
        out = make_image(w, h, c);
        *im = make_image(w, h, c);
    }
    return out;
}
// ----------------------------------------

int get_width_cv(mat_cv *ipl_src)
{
    IplImage *ipl = (IplImage *)ipl_src;
    return ipl->width;
}
// ----------------------------------------

int get_height_cv(mat_cv *ipl)
{
    //IplImage *ipl = (IplImage *)ipl_src;
    return ipl->height;
}
// ----------------------------------------

void release_ipl(mat_cv **ipl)
{
    IplImage **ipl_img = (IplImage **)ipl;
    if (*ipl_img) cvReleaseImage(ipl_img);
    *ipl_img = NULL;
}

// ====================================================================
// image-to-ipl, ipl-to-image
// ====================================================================
mat_cv *image_to_ipl(image im)
{
    int x, y, c;
    IplImage *disp = cvCreateImage(cvSize(im.w, im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for (y = 0; y < im.h; ++y) {
        for (x = 0; x < im.w; ++x) {
            for (c = 0; c < im.c; ++c) {
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return (mat_cv *)disp;
}
// ----------------------------------------

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for (i = 0; i < h; ++i) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < w; ++j) {
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.;
            }
        }
    }
    return im;
}
// ----------------------------------------

image ipl_to_image_custom(mat_cv* src)
{
    return ipl_to_image(src);
}
// ----------------------------------------

cv::Mat ipl_to_mat(IplImage *ipl)
{
    Mat m = cvarrToMat(ipl, true);
    return m;
}

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if (im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}
// ----------------------------------------

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image((mat_cv *)&ipl);
    rgbgr_image(im);
    return im;
}

// ====================================================================
// Window
// ====================================================================
void create_window_cv(char const* window_name, int full_screen, int width, int height)
{
    int window_type = WINDOW_NORMAL;
    if (full_screen) window_type = WINDOW_FULLSCREEN;

    cv::namedWindow(window_name, window_type);
    cv::moveWindow(window_name, 0, 0);
    cv::resizeWindow(window_name, width, height);
}
// ----------------------------------------

void destroy_all_windows_cv()
{
    cv::destroyAllWindows();
}
// ----------------------------------------

int wait_key_cv(int delay)
{
    return cv::waitKey(delay);
}
// ----------------------------------------

int wait_until_press_key_cv()
{
    return wait_key_cv(0);
}
// ----------------------------------------

void make_window(char *name, int w, int h, int fullscreen)
{
    cv::namedWindow(name, WINDOW_NORMAL);
    if (fullscreen) {
        cv::setWindowProperty(name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    }
    else {
        cv::resizeWindow(name, w, h);
        if (strcmp(name, "Demo") == 0) cv::moveWindow(name, 0, 0);
    }
}
// ----------------------------------------

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
// ----------------------------------------

void show_image_cv(image p, const char *name)
{
    int x, y, k;
    image copy = copy_image(p);
    constrain_image(copy);
    if (p.c == 3) rgbgr_image(copy);
    //normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL);
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    //++windows;
    for (y = 0; y < p.h; ++y) {
        for (x = 0; x < p.w; ++x) {
            for (k = 0; k < p.c; ++k) {
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
            }
        }
    }
    free_image(copy);
    if (0) {
        int w = 448;
        int h = w*p.h / p.w;
        if (h > 1000) {
            h = 1000;
            w = h*p.w / p.h;
        }
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_LINEAR);
        cvReleaseImage(&buffer);
    }
    cvShowImage(buff, disp);

    cvReleaseImage(&disp);
}
// ----------------------------------------

void show_image_cv_ipl(mat_cv *disp, const char *name)
{
    if (disp == NULL) return;
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);
    cv::namedWindow(buff, WINDOW_NORMAL);
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    //++windows;
    cvShowImage(buff, disp);
    //cvReleaseImage(&disp);
}


// ====================================================================
// Video Writer
// ====================================================================
write_cv *create_video_writer(char *out_filename, char c1, char c2, char c3, char c4, int fps, int width, int height, int is_color)
{
    try {
    cv::VideoWriter * output_video_writer =
        new cv::VideoWriter(out_filename, CV_FOURCC(c1, c2, c3, c4), fps, cv::Size(width, height), is_color);

    return (write_cv *)output_video_writer;
    }
    catch (...) {
        cerr << "OpenCV exception: create_video_writer \n";
    }
    return NULL;
}

void write_frame_cv(write_cv *output_video_writer, mat_cv *show_img)
{
    try {
        cv::VideoWriter *out = (cv::VideoWriter *)output_video_writer;
        out->write(ipl_to_mat(show_img));
    }
    catch (...) {
        cerr << "OpenCV exception: write_frame_cv \n";
    }
}

void release_video_writer(write_cv **output_video_writer)
{
    try {
        if (output_video_writer) {
            std::cout << " closing...";
            cv::VideoWriter *out = *(cv::VideoWriter **)output_video_writer;
            out->release();
            delete out;
            output_video_writer = NULL;
            std::cout << " closed!";
        }
        else {
            cerr << "OpenCV exception: output_video_writer isn't created \n";
        }
    }
    catch (...) {
        cerr << "OpenCV exception: release_video_writer \n";
    }
}

/*
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



int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}
*/


// ====================================================================
// Video Capture
// ====================================================================

cap_cv* get_capture_video_stream(const char *path) {
    cv::VideoCapture* cap = NULL;
    try {
        cap = new cv::VideoCapture(path);
    }
    catch (...) {
        cerr << " OpenCV exception: video-stream " << path << " can't be opened! \n";
    }
    return (cap_cv*)cap;
}
// ----------------------------------------

cap_cv* get_capture_webcam(int index)
{
    cv::VideoCapture* cap = NULL;
    try {
        cap = new cv::VideoCapture(index);
        //cap->set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        //cap->set(CV_CAP_PROP_FRAME_HEIGHT, 960);
    }
    catch (...) {
        cerr << " OpenCV exception: Web-camera " << index << " can't be opened! \n";
    }
    return (cap_cv*)cap;
}
// ----------------------------------------

void release_capture(cap_cv* cap)
{
    try {
        cv::VideoCapture *cpp_cap = (cv::VideoCapture *)cap;
        delete cpp_cap;
    }
    catch (...) {
        cerr << " OpenCV exception: cv::VideoCapture " << cap << " can't be released! \n";
    }
}
// ----------------------------------------

mat_cv* get_capture_frame_cv(cap_cv *cap) {
    IplImage* src = NULL;
    try {
        if (cap) {
            cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
            cv::Mat frame;
            if (cpp_cap.isOpened())
            {
                cpp_cap >> frame;
                IplImage tmp = frame;
                src = cvCloneImage(&tmp);
            }
            else std::cout << " Video-stream stopped! \n";
        }
        else cerr << " cv::VideoCapture isn't created \n";
    }
    catch (...) {
        std::cout << " OpenCV exception: Video-stream stoped! \n";
    }
    return (mat_cv *)src;
}
// ----------------------------------------

int get_stream_fps_cpp_cv(cap_cv *cap)
{
    int fps = 25;
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        fps = cpp_cap.get(CAP_PROP_FPS);
#else                        // OpenCV 2.x
        fps = cpp_cap.get(CV_CAP_PROP_FPS);
#endif
    }
    catch (...) {
        cerr << " Can't get FPS of source videofile. For output video FPS = 25 by default. \n";
    }
    return fps;
}
// ----------------------------------------

double get_capture_property_cv(cap_cv *cap, int property_id)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        return cpp_cap.get(property_id);
    }
    catch (...) {
        cerr << " OpenCV exception: Can't get property of source video-stream. \n";
    }
    return 0;
}
// ----------------------------------------

double get_capture_frame_count_cv(cap_cv *cap)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        return cpp_cap.get(CAP_PROP_FRAME_COUNT);
#else                        // OpenCV 2.x
        return cpp_cap.get(CV_CAP_PROP_FRAME_COUNT);
#endif
    }
    catch (...) {
        cerr << " OpenCV exception: Can't get CAP_PROP_FRAME_COUNT of source videofile. \n";
    }
    return 0;
}
// ----------------------------------------

int set_capture_property_cv(cap_cv *cap, int property_id, double value)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        return cpp_cap.set(property_id, value);
    }
    catch (...) {
        cerr << " Can't set property of source video-stream. \n";
    }
    return false;
}
// ----------------------------------------

int set_capture_position_frame_cv(cap_cv *cap, int index)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        return cpp_cap.set(CAP_PROP_POS_FRAMES, index);
#else                        // OpenCV 2.x
        return cpp_cap.set(CV_CAP_PROP_POS_FRAMES, index);
#endif
    }
    catch (...) {
        cerr << " Can't set CAP_PROP_POS_FRAMES of source videofile. \n";
    }
    return false;
}
// ----------------------------------------



// ====================================================================
// ... Video Capture
// ====================================================================

image get_image_from_stream_cpp(cap_cv *cap)
{
    IplImage* src;
    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->width < 1 || src->height < 1 || src->nChannels < 1);
        printf("Video stream: %d x %d \n", src->width, src->height);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!src) return make_empty_image(0, 0, 0);
    image im = ipl_to_image(src);
    rgbgr_image(im);
    return im;
}
// ----------------------------------------

int wait_for_stream(cap_cv *cap, IplImage* src, int dont_close)
{
    if (!src) {
        if (dont_close) src = cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        else return 0;
    }
    if (src->width < 1 || src->height < 1 || src->nChannels < 1) {
        if (dont_close) {
            cvReleaseImage(&src);
            int z = 0;
            for (z = 0; z < 20; ++z) {
                get_capture_frame_cv(cap);
                cvReleaseImage(&src);
            }
            src = cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        }
        else return 0;
    }
    return 1;
}
// ----------------------------------------

image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
    c = c ? c : 3;
    IplImage* src;

    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->width < 1 || src->height < 1 || src->nChannels < 1);
        printf("Video stream: %d x %d \n", src->width, src->height);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);
    IplImage* new_img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, c);
    *in_img = (mat_cv *)cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, c);
    cvResize(src, *in_img, CV_INTER_LINEAR);
    cvResize(src, new_img, CV_INTER_LINEAR);
    image im = ipl_to_image(new_img);
    cvReleaseImage(&new_img);
    cvReleaseImage(&src);
    if (c>1)
        rgbgr_image(im);
    return im;
}
// ----------------------------------------

image get_image_from_stream_letterbox(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
    c = c ? c : 3;
    IplImage* src;
    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->width < 1 || src->height < 1 || src->nChannels < 1);
        printf("Video stream: %d x %d \n", src->width, src->height);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);
    *in_img = (mat_cv *)cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, c);
    cvResize(src, *in_img, CV_INTER_LINEAR);
    image tmp = ipl_to_image(src);
    image im = letterbox_image(tmp, w, h);
    free_image(tmp);
    cvReleaseImage(&src);
    if (c>1) rgbgr_image(im);
    return im;
}
// ----------------------------------------

// ====================================================================
// Image Saving
// ====================================================================
extern int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);
extern int stbi_write_jpg(char const *filename, int x, int y, int comp, const void  *data, int quality);

void save_cv_png(mat_cv *img_src, const char *name)
{
    IplImage* img = (IplImage* )img_src;
    IplImage* img_rgb = cvCreateImage(cvSize(img->width, img->height), 8, 3);
    cvCvtColor(img, img_rgb, CV_RGB2BGR);
    stbi_write_png(name, img_rgb->width, img_rgb->height, 3, (char *)img_rgb->imageData, 0);
    cvRelease((void**)&img_rgb);
}
// ----------------------------------------

void save_cv_jpg(mat_cv *img_src, const char *name)
{
    IplImage* img = (IplImage*)img_src;
    IplImage* img_rgb = cvCreateImage(cvSize(img->width, img->height), 8, 3);
    cvCvtColor(img, img_rgb, CV_RGB2BGR);
    stbi_write_jpg(name, img_rgb->width, img_rgb->height, 3, (char *)img_rgb->imageData, 80);
    cvRelease((void**)&img_rgb);
}
// ----------------------------------------

// ====================================================================
// Draw Detection
// ====================================================================
void draw_detections_cv_v3(mat_cv* show_img, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    int i, j;
    if (!show_img) return;
    static int frame_id = 0;
    frame_id++;

    for (i = 0; i < num; ++i) {
        char labelstr[4096] = { 0 };
        int class_id = -1;
        for (j = 0; j < classes; ++j) {
            int show = strncmp(names[j], "dont_show", 9);
            if (dets[i].prob[j] > thresh && show) {
                if (class_id < 0) {
                    strcat(labelstr, names[j]);
                    class_id = j;
                }
                else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%% ", names[j], dets[i].prob[j] * 100);
            }
        }
        if (class_id >= 0) {
            int width = show_img->height * .006;

            //if(0){
            //width = pow(prob, 1./2.)*10+1;
            //alphabet = 0;
            //}

            //printf("%d %s: %.0f%%\n", i, names[class_id], prob*100);
            int offset = class_id * 123457 % classes;
            float red = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue = get_color(0, offset, classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;
            b.w = (b.w < 1) ? b.w : 1;
            b.h = (b.h < 1) ? b.h : 1;
            b.x = (b.x < 1) ? b.x : 1;
            b.y = (b.y < 1) ? b.y : 1;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left = (b.x - b.w / 2.)*show_img->width;
            int right = (b.x + b.w / 2.)*show_img->width;
            int top = (b.y - b.h / 2.)*show_img->height;
            int bot = (b.y + b.h / 2.)*show_img->height;

            if (left < 0) left = 0;
            if (right > show_img->width - 1) right = show_img->width - 1;
            if (top < 0) top = 0;
            if (bot > show_img->height - 1) bot = show_img->height - 1;

            //int b_x_center = (left + right) / 2;
            //int b_y_center = (top + bot) / 2;
            //int b_width = right - left;
            //int b_height = bot - top;
            //sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

            float const font_size = show_img->height / 1000.F;
            CvPoint pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
            pt1.x = left;
            pt1.y = top;
            pt2.x = right;
            pt2.y = bot;
            pt_text.x = left;
            pt_text.y = top - 12;
            pt_text_bg1.x = left;
            pt_text_bg1.y = top - (10 + 25 * font_size);
            pt_text_bg2.x = right;
            pt_text_bg2.y = top;
            CvScalar color;
            color.val[0] = red * 256;
            color.val[1] = green * 256;
            color.val[2] = blue * 256;

            // you should create directory: result_img
            //static int copied_frame_id = -1;
            //static IplImage* copy_img = NULL;
            //if (copied_frame_id != frame_id) {
            //    copied_frame_id = frame_id;
            //    if(copy_img == NULL) copy_img = cvCreateImage(cvSize(show_img->width, show_img->height), show_img->depth, show_img->nChannels);
            //    cvCopy(show_img, copy_img, 0);
            //}
            //static int img_id = 0;
            //img_id++;
            //char image_name[1024];
            //sprintf(image_name, "result_img/img_%d_%d_%d_%s.jpg", frame_id, img_id, class_id, names[class_id]);
            //CvRect rect = cvRect(pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
            //cvSetImageROI(copy_img, rect);
            //cvSaveImage(image_name, copy_img, 0);
            //cvResetImageROI(copy_img);

            cvRectangle(show_img, pt1, pt2, color, width, 8, 0);
            if (ext_output)
                printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                (float)left, (float)top, b.w*show_img->width, b.h*show_img->height);
            else
                printf("\n");

            cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
            cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);    // filled
            CvScalar black_color;
            black_color.val[0] = 0;
            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, font_size, font_size, 0, font_size * 3, 8);
            cvPutText(show_img, labelstr, pt_text, &font, black_color);
        }
    }
    if (ext_output) {
        fflush(stdout);
    }
}
// ----------------------------------------

// ====================================================================
// Draw Loss & Accuracy chart
// ====================================================================
mat_cv* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size, int dont_show)
{
    int img_offset = 50;
    int draw_size = img_size - img_offset;
    IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
    cvSet(img, CV_RGB(255, 255, 255), 0);
    CvPoint pt1, pt2, pt_text;
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 1, CV_AA);
    char char_buff[100];
    int i;
    // vertical lines
    pt1.x = img_offset; pt2.x = img_size, pt_text.x = 10;
    for (i = 1; i <= number_of_lines; ++i) {
        pt1.y = pt2.y = (float)i * draw_size / number_of_lines;
        cvLine(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
        if (i % 10 == 0) {
            sprintf(char_buff, "%2.1f", max_img_loss*(number_of_lines - i) / number_of_lines);
            pt_text.y = pt1.y + 5;
            cvPutText(img, char_buff, pt_text, &font, CV_RGB(0, 0, 0));
            cvLine(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
        }
    }
    // horizontal lines
    pt1.y = draw_size; pt2.y = 0, pt_text.y = draw_size + 15;
    for (i = 0; i <= number_of_lines; ++i) {
        pt1.x = pt2.x = img_offset + (float)i * draw_size / number_of_lines;
        cvLine(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
        if (i % 10 == 0) {
            sprintf(char_buff, "%d", max_batches * i / number_of_lines);
            pt_text.x = pt1.x - 20;
            cvPutText(img, char_buff, pt_text, &font, CV_RGB(0, 0, 0));
            cvLine(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
        }
    }

    cvPutText(img, "Loss", cvPoint(0, 35), &font, CV_RGB(0, 0, 255));
    cvPutText(img, "Iteration number", cvPoint(draw_size / 2, img_size - 10), &font, CV_RGB(0, 0, 0));
    char max_batches_buff[100];
    sprintf(max_batches_buff, "in cfg max_batches=%d", max_batches);
    cvPutText(img, max_batches_buff, cvPoint(draw_size - 195, img_size - 10), &font, CV_RGB(0, 0, 0));
    cvPutText(img, "Press 's' to save: chart.png", cvPoint(5, img_size - 10), &font, CV_RGB(0, 0, 0));
    if (!dont_show) {
        printf(" If error occurs - run training with flag: -dont_show \n");
        cvNamedWindow("average loss", CV_WINDOW_NORMAL);
        cvMoveWindow("average loss", 0, 0);
        cvResizeWindow("average loss", img_size, img_size);
        cvShowImage("average loss", img);
        cvWaitKey(20);
    }
    return (mat_cv*)img;
}
// ----------------------------------------

void draw_train_loss(mat_cv* img_src, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches,
    float precision, int draw_precision, char *accuracy_name, int dont_show, int mjpeg_port)
{
    IplImage* img = (IplImage*)img_src;
    int img_offset = 50;
    int draw_size = img_size - img_offset;
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 1, CV_AA);
    char char_buff[100];
    CvPoint pt1, pt2;
    pt1.x = img_offset + draw_size * (float)current_batch / max_batches;
    pt1.y = draw_size * (1 - avg_loss / max_img_loss);
    if (pt1.y < 0) pt1.y = 1;
    cvCircle(img, pt1, 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);

    // precision
    if (draw_precision) {
        static float old_precision = 0;
        static int iteration_old = 0;
        static int text_iteration_old = 0;
        if (iteration_old == 0) cvPutText(img, accuracy_name, cvPoint(0, 12), &font, CV_RGB(255, 0, 0));

        cvLine(img,
            cvPoint(img_offset + draw_size * (float)iteration_old / max_batches, draw_size * (1 - old_precision)),
            cvPoint(img_offset + draw_size * (float)current_batch / max_batches, draw_size * (1 - precision)),
            CV_RGB(255, 0, 0), 1, 8, 0);

        if (((int)(old_precision * 10) != (int)(precision * 10)) || (current_batch - text_iteration_old) >= max_batches / 10) {
            text_iteration_old = current_batch;
            sprintf(char_buff, "%2.0f%% ", precision * 100);
            CvFont font3;
            cvInitFont(&font3, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 5, CV_AA);
            cvPutText(img, char_buff, cvPoint(pt1.x - 30, draw_size * (1 - precision) + 15), &font3, CV_RGB(255, 255, 255));

            CvFont font2;
            cvInitFont(&font2, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 1, CV_AA);
            cvPutText(img, char_buff, cvPoint(pt1.x - 30, draw_size * (1 - precision) + 15), &font2, CV_RGB(200, 0, 0));
        }
        old_precision = precision;
        iteration_old = current_batch;
    }

    sprintf(char_buff, "current avg loss = %2.4f    iteration = %d", avg_loss, current_batch);
    pt1.x = 55, pt1.y = 10;
    pt2.x = pt1.x + 460, pt2.y = pt1.y + 20;
    cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
    pt1.y += 15;
    cvPutText(img, char_buff, pt1, &font, CV_RGB(0, 0, 0));

    int k = 0;
    if (!dont_show) {
        cvShowImage("average loss", img);
        k = cvWaitKey(20);
    }
    if (k == 's' || current_batch == (max_batches - 1) || current_batch % 100 == 0) {
        save_cv_png((mat_cv *)img, "chart.png");
        cvPutText(img, "- Saved", cvPoint(250, img_size - 10), &font, CV_RGB(255, 0, 0));
    }
    else
        cvPutText(img, "- Saved", cvPoint(250, img_size - 10), &font, CV_RGB(255, 255, 255));

    if (mjpeg_port > 0) send_mjpeg((mat_cv *)img, mjpeg_port, 500000, 100);
}
// ----------------------------------------


// ====================================================================
// Data augmentation
// ====================================================================
image image_data_augmentation(mat_cv* ipl, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float jitter, float dhue, float dsat, float dexp)
{
    image out;
    try {
        cv::Mat img = cv::cvarrToMat(ipl);

        // crop
        cv::Rect src_rect(pleft, ptop, swidth, sheight);
        cv::Rect img_rect(cv::Point2i(0, 0), img.size());
        cv::Rect new_src_rect = src_rect & img_rect;

        cv::Rect dst_rect(cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)), new_src_rect.size());

        cv::Mat cropped(cv::Size(src_rect.width, src_rect.height), img.type());
        cropped.setTo(cv::Scalar::all(0));

        img(new_src_rect).copyTo(cropped(dst_rect));

        // resize
        cv::Mat sized;
        cv::resize(cropped, sized, cv::Size(w, h), 0, 0, INTER_LINEAR);

        // flip
        if (flip) {
            cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
            sized = cropped.clone();
        }

        // HSV augmentation
        // CV_BGR2HSV, CV_RGB2HSV, CV_HSV2BGR, CV_HSV2RGB
        if (ipl->nChannels >= 3)
        {
            cv::Mat hsv_src;
            cvtColor(sized, hsv_src, CV_BGR2HSV);    // also BGR -> RGB

            std::vector<cv::Mat> hsv;
            cv::split(hsv_src, hsv);

            hsv[1] *= dsat;
            hsv[2] *= dexp;
            hsv[0] += 179 * dhue;

            cv::merge(hsv, hsv_src);

            cvtColor(hsv_src, sized, CV_HSV2RGB);    // now RGB instead of BGR
        }
        else
        {
            sized *= dexp;
        }

        //std::stringstream window_name;
        //window_name << "augmentation - " << ipl;
        //cv::imshow(window_name.str(), sized);
        //cv::waitKey(0);

        // Mat -> IplImage -> image
        IplImage src = sized;
        out = ipl_to_image(&src);
    }
    catch (...) {
        cerr << "OpenCV can't augment image: " << w << " x " << h << " \n";
        out = ipl_to_image(ipl);
    }
    return out;
}

// ====================================================================
// Show Anchors
// ====================================================================
void show_acnhors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height)
{
    CvMat* labels = cvCreateMat(number_of_boxes, 1, CV_32SC1);
    CvMat* points = cvCreateMat(number_of_boxes, 2, CV_32FC1);
    CvMat* centers = cvCreateMat(num_of_clusters, 2, CV_32FC1);

    int i, j;
    for (i = 0; i < number_of_boxes; ++i) {
        points->data.fl[i * 2] = rel_width_height_array[i * 2];
        points->data.fl[i * 2 + 1] = rel_width_height_array[i * 2 + 1];
        //cvSet1D(points, i * 2, cvScalar(rel_width_height_array[i * 2], 0, 0, 0));
        //cvSet1D(points, i * 2 + 1, cvScalar(rel_width_height_array[i * 2 + 1], 0, 0, 0));
    }

    for (i = 0; i < num_of_clusters; ++i) {
        centers->data.fl[i * 2] = anchors_data.centers.vals[i][0];
        centers->data.fl[i * 2 + 1] = anchors_data.centers.vals[i][1];
    }

    for (i = 0; i < number_of_boxes; ++i) {
        labels->data.i[i] = anchors_data.assignments[i];
    }

    size_t img_size = 700;
    IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
    cvZero(img);
    for (i = 0; i < number_of_boxes; ++i) {
        CvPoint pt;
        pt.x = points->data.fl[i * 2] * img_size / width;
        pt.y = points->data.fl[i * 2 + 1] * img_size / height;
        int cluster_idx = labels->data.i[i];
        int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
        int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
        int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
        cvCircle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
        //if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
    }

    for (j = 0; j < num_of_clusters; ++j) {
        CvPoint pt1, pt2;
        pt1.x = pt1.y = 0;
        pt2.x = centers->data.fl[j * 2] * img_size / width;
        pt2.y = centers->data.fl[j * 2 + 1] * img_size / height;
        cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
    }
    save_cv_png((mat_cv *)img, "cloud.png");
    cvShowImage("clusters", img);
    cvWaitKey(0);
    cvReleaseImage(&img);
    cvDestroyAllWindows();
    cvReleaseMat(&labels);
    cvReleaseMat(&points);
    cvReleaseMat(&centers);
}

}   // extern "C"


#else  // OPENCV
int wait_key_cv(int delay) { return 0; }
int wait_until_press_key_cv() { return 0; }
void destroy_all_windows_cv() {}
#endif // OPENCV
