#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <sys/time.h>
}

/* Change class number here */
#define CLS_NUM 2

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern "C" IplImage* image_to_Ipl(image img, int w, int h, int depth, int c, int step);
extern "C" image ipl_to_image(IplImage* src);
extern "C" void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern "C" void draw_yolo(image im, int num, float thresh, box *boxes, float **probs);

extern "C" char *voc_names[];
extern "C" image voc_labels[];

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp ;
static cv::VideoCapture cap;
static cv::VideoWriter cap_out;
static float fps = 0;
static float demo_thresh = 0;
static int w, h, depth, c, step= 0;
static int MODE = -1;

void *fetch_in_thread(void *ptr)
{
    cv::Mat frame_m;
    cap >> frame_m;
    IplImage frame = frame_m;

if(step == 0)
{
    w = frame.width;
    h = frame.height;
    c = frame.nChannels;
    depth= frame.depth; 
    step = frame.widthStep;
}

    in = ipl_to_image(&frame);
    rgbgr_image(in);
    in_s = resize_image(in, net.w, net.h);
    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    detection_layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *predictions = network_predict(net, X);
    free_image(det_s);
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.0f\n",fps);
    printf("Objects:\n\n");
    draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, CLS_NUM);

    if(MODE == 1)
    {
        IplImage* outputIpl= image_to_Ipl(det, w, h, depth, c, step);
        cv::Mat outputMat = cv::cvarrToMat(outputIpl, true);
        /*
        cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
        cvShowImage("image", outputIpl); 
        cvWaitKey(1);  
        */
        cvReleaseImage(&outputIpl);
        cap_out << outputMat;
        outputMat.release();
     }

    return 0;
}

extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *videofile)
{
    demo_thresh = thresh;
    printf("YOLO demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

if(cam_index != -1)
{
    MODE = 0; 
    cv::VideoCapture cam(cam_index);
    cap = cam;
    if(!cap.isOpened()) error("Couldn't connect to webcam.\n");
}
else 
{
    MODE = 1;
    printf("Video File name is: %s\n", videofile);
    cv::VideoCapture videoCap(videofile);
    cap = videoCap;
    if(!cap.isOpened()) error("Couldn't read video file.\n");

    cv::Size S = cv::Size((int)videoCap.get(CV_CAP_PROP_FRAME_WIDTH), (int)videoCap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter outputVideo("out.avi", CV_FOURCC('D','I','V','X'), videoCap.get(CV_CAP_PROP_FPS), S, true);
    if(!outputVideo.isOpened()) error("Couldn't write video file.\n");
    cap_out = outputVideo;
}
 
    detection_layer l = net.layers[net.n-1];
    int j;

    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        show_image(disp, "YOLO");
        free_image(disp);
        cvWaitKey(1);
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);

        disp  = det;
        det   = in;
        det_s = in_s;

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
}
#else
extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index){
    fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
#endif
