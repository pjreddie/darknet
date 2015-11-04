extern "C" {
#include "network.h"
#include "region_layer.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
}

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern "C" image ipl_to_image(IplImage* src);
extern "C" void convert_swag_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern "C" void draw_swag(image im, int num, float thresh, box *boxes, float **probs, char *label);

extern "C" void demo_swag(char *cfgfile, char *weightfile, float thresh)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    region_layer l = net.layers[net.n-1];
    cv::VideoCapture cap(0);

    set_batch_network(&net, 1);
    srand(2222222);
    float nms = .4;
    int j;
    box *boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    while(1){
        cv::Mat frame_m;
        cap >> frame_m;
        IplImage frame = frame_m;
        image im = ipl_to_image(&frame);
        rgbgr_image(im);

        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        float *predictions = network_predict(net, X);
        convert_swag_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nObjects:\n\n");
        draw_swag(im, l.side*l.side*l.n, thresh, boxes, probs, "predictions");

        free_image(im);
        free_image(sized);
        cvWaitKey(1);
    }
}
#else
extern "C" void demo_swag(char *cfgfile, char *weightfile, float thresh){}
#endif

