//
// Created by frivas on 25/04/16.
//

#include "DarknetAPI.h"
#include <iostream>
#include "DarknetAPIConversions.h"

//char *voc_names2[] = {"gable_main", "hip_main", "hip_ext", "hipgable_main", "hipgable_ext",
//                              "nested_turret"};
//image voc_labels[6];


char *voc_names2[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
image voc_labels[20];




void addDetection(image& im, int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes, DarknetDetections& detections){

    int i;
    for(i = 0; i < num; ++i){
        int classid = c_max_index(probs[i], classes);
        float prob = probs[i][classid];
        if(prob > thresh){
            int width = pow(prob, 1./2.)*10+1;
            int offset = classid*17 % classes;
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            box detectedbox;

            detectedbox.x = left;
            detectedbox.y = top;
            detectedbox.h = bot - top;
            detectedbox.w = right - left;

//            std::cout<<"---------------------:" << detectedbox.x<<" "<<detectedbox.y<<" "<<detectedbox.w<<" "<<detectedbox.h<<std::endl;
            DarknetDetection detection(detectedbox,classid,prob);

            detections.push_back(detection);


        }
    }
}


DarknetDetections processImageDetection(network& net,image& im, float thresh ){

    float hier_thresh=0.5;

    DarknetDetections detections;

    c_set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    int j;
    float nms=.4;

    image sized = c_resize_image(im, net.w, net.h);
    layer l = net.layers[net.n-1];

    box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] =(float *) calloc(l.classes + 1, sizeof(float *));

    float **masks = 0;
    if (l.coords > 4){
        masks = (float **)calloc(l.w*l.h*l.n, sizeof(float*));
        for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
    }

    float *X = sized.data;
    time=clock();
    c_network_predict(net, X);
    printf("Predicted in %f seconds.\n", c_sec(clock()-time));
//    c_get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
    c_get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
    if (l.softmax_tree && nms) c_do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    else if (nms) c_do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    addDetection( im, l.w*l.h * l.n, thresh, boxes, probs, voc_names2, 0, l.classes, detections );
    c_free_image(sized);

    return detections;

}


DarknetDetections processImageDetection(network& net,const  cv::Mat & im, float thresh){
    image imDark= cv_to_image(im);
    return processImageDetection(net,imDark,thresh);
}



DarknetAPI::DarknetAPI(char *cfgfile, char *weightfile) {
    net = c_parse_network_cfg(cfgfile);
    if (weightfile) {
        c_load_weights(&net, weightfile);
    }

    fprintf(stderr, "4 \n");

}
DarknetAPI::~DarknetAPI(){
     c_free_network(net);
}


DarknetDetections DarknetAPI::process(image& im, float thresh ){
    return processImageDetection(this->net,im,thresh);
}

DarknetDetections DarknetAPI::process(const cv::Mat &im, float thresh) {
    image imDark= cv_to_image(im);
    return processImageDetection(this->net,imDark,thresh);
}

std::string DarknetAPI::processToJson(const cv::Mat &im, float thresh) {
    return process(im,thresh).serialize();
}
