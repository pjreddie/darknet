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

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern "C" image ipl_to_image(IplImage* src);
extern "C" void convert_coco_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);

extern "C" char *coco_classes[];
extern "C" image coco_labels[];

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp ;
static cv::VideoCapture cap;
static float fps = 0;
static float demo_thresh = 0;

static const int frames = 3;
static float *predictions[frames];
static int demo_index = 0;
static image images[frames];
static float *avg;

void *fetch_in_thread_coco(void *ptr)
{
    cv::Mat frame_m;
    cap >> frame_m;
    IplImage frame = frame_m;
    in = ipl_to_image(&frame);
    rgbgr_image(in);
    in_s = resize_image(in, net.w, net.h);
    return 0;
}

void *detect_in_thread_coco(void *ptr)
{
    float nms = .4;

    detection_layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, frames, l.outputs, avg);

    free_image(det_s);
    convert_coco_detections(avg, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.0f\n",fps);
    printf("Objects:\n\n");

    images[demo_index] = det;
    det = images[(demo_index + frames/2 + 1)%frames];
    demo_index = (demo_index + 1)%frames;

    draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes, probs, coco_classes, coco_labels, 80);
    return 0;
}

extern "C" void demo_coco(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename)
{
    demo_thresh = thresh;
    printf("YOLO demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

    if(filename){
        cap.open(filename);
    }else{
        cap.open(cam_index);
    }

    if(!cap.isOpened()) error("Couldn't connect to webcam.\n");

    detection_layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < frames; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < frames; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread_coco(0);
    det = in;
    det_s = in_s;

    fetch_in_thread_coco(0);
    detect_in_thread_coco(0);
    disp = det;
    det = in;
    det_s = in_s;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);
        if(pthread_create(&fetch_thread, 0, fetch_in_thread_coco, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread_coco, 0)) error("Thread creation failed");
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
extern "C" void demo_coco(char *cfgfile, char *weightfile, float thresh, int cam_index){
    fprintf(stderr, "YOLO-COCO demo needs OpenCV for webcam images.\n");
}
#endif

