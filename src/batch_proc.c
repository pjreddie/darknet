#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

static void save_detections(FILE * fout,image im, int id, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes,const char * name)
{
    int i;

    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){


            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            //printf("%s: %.0f%%\n", names[class], prob*100);
            
            //width = prob*20+2;

            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;


            fprintf(fout,"%d\t%d\t%f\t%d\t%d\t%d\t%d\t%s\n",id,class,prob,left,top,right,bot,name);
        }
    }
    if((id % 20) == 0)
    {
        fflush(fout);
    }
}

static double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


static FILE * fout = 0;
static int demo_delay = 0;
static int demo_frame = 3;
static float **predictions;
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static float demo_thresh = 0;
static float demo_hier = .5;
static float **probs;
static box *boxes;
static network net;
static int demo_detections = 0;
static float *avg;

static double demo_time;

static network net2;
static float **probs2;
static box *boxes2;
static float **predictions2;
static image buff [3];
static image buff_letter[3];
static int buff_fetch_index = 0;
static int buff_detect_index = 0;
static int buff_show_index = 0;
static float *last_avg2;
static float *last_avg;

#ifdef OPENCV


static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static int running = 0;
static int demo_index = 0;
static int demo_done = 0;
static int overall_frame = 0;
static CvVideoWriter * writer = 0;


static void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = buff_letter[buff_detect_index].data; // was +2
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, buff[0].w, buff[0].h, net.w, net.h, demo_thresh, probs, boxes, 0, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Frame: %d\n",overall_frame);
    printf("Objects:\n\n");

    image display = buff[buff_detect_index]; // was +2
    draw_detections(display, demo_detections, demo_thresh, boxes, probs, 0, demo_names, demo_alphabet, demo_classes);
    if(fout)
    {
        // static void save_detections(FILE * fout,image im, int id, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes,const char * name)
        save_detections(fout,display, overall_frame,demo_detections, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes,"");
    }
    overall_frame++;

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

static void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_fetch_index]);
    letterbox_image_into(buff[buff_fetch_index], net.w, net.h, buff_letter[buff_fetch_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

static void *display_in_thread(void *ptr)
{
    show_image_cv(buff[buff_show_index], "Demo", ipl); // was +1
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

static void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

static void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

static int VideoWriter_fourcc(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}


static void dowrite(image im, const char * voutput)
{
    if(!writer)
    {
        const char * rf = strchr(voutput,':');
        int fourcc = 0;
        CvSize xsize;
        if(rf)
        {
            voutput = rf+1;
            fourcc = VideoWriter_fourcc(voutput[0],voutput[1],voutput[2],voutput[3]);
        }
        xsize.width = im.w;
        xsize.height = im.h;
        writer = cvCreateVideoWriter(voutput,fourcc,25,xsize,0);
        if(!writer)
        {
            fprintf(stderr,"cannot save file %s with forucc %d\n",voutput,fourcc);
            exit(1);
        }
    }

    // sloooooow
    {
        if(im.c == 3) rgbgr_image(im);
        int step = ipl->widthStep;
        int x, y, k;
        for(y = 0; y < im.h; ++y){
            for(x = 0; x < im.w; ++x){
                for(k= 0; k < im.c; ++k){
                    ipl->imageData[y*step + x*im.c + k] = (unsigned char)(get_pixel(im,x,y,k)*255);
                }
            }
        }
        cvWriteFrame(writer,ipl);
    }
}
#endif


image load_image(char *filename, int w, int h, int c);
image load_image_stb(char *filename, int channels);
image letterbox_image(image im, int w, int h);

void batch_processor(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, const char * output, const char * voutput, int allframes)
{
    fprintf(stderr,"allocation of predictions\n");
    predictions = calloc(1, sizeof(float*));
    if(!predictions)
        return;
    fprintf(stderr,"loading alphabet\n");
    image **alphabet = load_alphabet();
    if(!alphabet)
        return;
    fprintf(stderr,"loading network\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        fprintf(stderr,"loading weights\n");
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    int overall_frame = 1;

    srand(2222222);

#ifndef OPENCV
    if(!filename) {
        error("only filenames in non OpenCV mode");
        return;
    }
#endif
    fprintf(stderr,"batch processor ready to start\n");

    layer l = net.layers[net.n-1];
    int j;

    predictions[0] = (float *) calloc(l.outputs, sizeof(float));
    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));


#ifdef OPENCV
    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }
    if(!cap) error("Couldn't connect to webcam or input file\n");
#endif
    FILE* file = fopen(filename, "r"); /* should check the result */
    if(!file)
        return;

    // output file with the bounding boxes
    if(output)
    {
        fout = fopen(output,"w");
        if(!fout)
        {
            fprintf(stderr,"cannot write output %s\n",output);
        }
        fprintf(fout,"frame\tclass\tprob\tleft\ttop\tright\tbottom\tfilename\n");
        fflush(fout);
    }

    char item_filename[256];
    float nms = .4;
#ifdef OPENCV
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }
#endif

    //buff_letter[1] = make_empty_image(net.w,net.h,3);
        
#ifdef OPENCV
    while (1) {
        buff[0] = get_image_from_stream(cap);
#else
    while (fgets(item_filename, sizeof(item_filename), file)) {
        item_filename[strlen(item_filename)-1] = 0;
        buff[0] = load_image_stb(item_filename,3);
#endif
        if(overall_frame == 1)
        {
            buff_letter[0] =  letterbox_image(buff[0], net.w, net.h);
        }
        else
        {              
            letterbox_image_into(buff[0] , net.w, net.h,buff_letter[0]);
        }
        fprintf(stderr,"%10d read file: %s for input %d %d\n",overall_frame, item_filename,net.w,net.h);

        layer l = net.layers[net.n-1]; // last layer
        float *X = buff_letter[0].data; // input data
        float *prediction = network_predict(net, X);

        memcpy(predictions[0], prediction, l.outputs*sizeof(float));
        //mean_arrays(predictions, demo_frame, l.outputs, avg);
        l.output = predictions[0];
        fprintf(stderr,"prediction done then...\n");
        if(l.type == DETECTION){
            fprintf(stderr,"get_detection_boxes\n");
            get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
        } else if (l.type == REGION){
            fprintf(stderr,"get_region_boxes \n");
            get_region_boxes(l, buff[0].w, buff[0].h, net.w, net.h, thresh, probs, boxes, 0, 0, 0, hier, 1);
        } else {
            error("Last layer must produce detections\n");
        }
        if (nms > 0) {
            fprintf(stderr,"nms \n");
            do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        }


        if(fout)
        {
            fprintf(stderr,"save_detections \n");
            // image frameidentifier numberofdetections
            save_detections(fout,buff[0], overall_frame,l.n*l.w*l.h, thresh, boxes, probs, names, alphabet, classes,item_filename);
        }

#ifdef OPENCV
        show_image_cv(buff[buff_show_index], "Demo", ipl); // was +1
        if(voutput)
        {
            dowrite(buff[buff_show_index],voutput);  // was +1
        }
#endif

        overall_frame++;
    }
    
    if(fout)
    {
        fclose(fout);
        fout = 0;
    }

#ifdef OPENCV
    if(writer)
    {
        cvReleaseVideoWriter(&writer);
        writer = 0;
    }
#else
    fclose(file);

#endif


}

