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

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image buff [3];
static image buff_letter[3];
static int buff_fetch_index = 0;
static int buff_detect_index = 0;
static int buff_show_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_delay = 0;
static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static int overall_frame = 0;
static float *last_avg2;
static float *last_avg;
static float *avg;
static CvVideoWriter * writer = 0;
static FILE * fout = 0;

double demo_time;

void save_detections(image im, int id, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
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


            fprintf(fout,"%d\t%d\t%f\t%d\t%d\t%d\t%d\n",id,class,prob,left,top,right,bot);
        }
    }
    if((id % 20) == 0)
    {
        fflush(fout);
    }
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = buff_letter[buff_detect_index].data; // was +2
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = last_avg2;
    if(demo_delay == 0) l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
        printf("!DE\n");
    } else if (l.type == REGION){
        get_region_boxes(l, buff[0].w, buff[0].h, net.w, net.h, demo_thresh, probs, boxes, 0, 0, demo_hier, 1);
        printf("!RE\n");
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Frame: %d\n",overall_frame);
    printf("Objects:\n\n");
    image display = buff[buff_show_index]; // was +2
    draw_detections(display, demo_detections, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
    if(fout)
        save_detections(display, overall_frame,demo_detections, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
    overall_frame++;
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_fetch_index]);
    letterbox_image_into(buff[buff_fetch_index], net.w, net.h, buff_letter[buff_fetch_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[buff_show_index], "Demo", ipl); // was +1
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 10){
        if(demo_delay == 0) demo_delay = 60;
        else if(demo_delay == 5) demo_delay = 0;
        else if(demo_delay == 60) demo_delay = 5;
        else demo_delay = 0;
    } else if (c == 27) {
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

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

static int VideoWriter_fourcc(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}


void dowrite(image im, const char * voutput)
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

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen,const char * output, const char * voutput, int allframes)
{
    demo_delay = delay;
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

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

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    last_avg  = (float *) calloc(l.outputs, sizeof(float));
    last_avg2 = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[1] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[2] = letterbox_image(buff[0], net.w, net.h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = get_wall_time();

    if(output)
    {
        fout = fopen(output,"w");
        if(!fout)
        {
            fprintf(stderr,"cannot write output %s\n",output);
        }
        fprintf(fout,"frame\tclass\tprob\tleft\ttop\tright\tbottom\n");
        fflush(fout);
    }

    int buff_index = 0;
    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(allframes)
        {
            buff_fetch_index = buff_index;
            buff_detect_index = buff_index;
            buff_show_index = buff_index;
            fetch_in_thread(0);
            detect_in_thread(0);
        }
        else
        {            
            buff_fetch_index = buff_index;
            buff_show_index = (buff_index+1)%3;
            buff_detect_index = (buff_index+2)%3;
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");            
        }

        // always timing stats
        if(count % (demo_delay+1) == 0){
            fps = 1./(get_wall_time() - demo_time);
            demo_time = get_wall_time();
            float *swap = last_avg;
            last_avg  = last_avg2;
            last_avg2 = swap;
            memcpy(last_avg, avg, l.outputs*sizeof(float));
        }

        if(!prefix){
            display_in_thread(0);            
        }else if(prefix[0] != '!') { 
            // prefix ! for skipping save and visualization
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[buff_show_index], name); // was +1
        }            
        if(!allframes)
        {
            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);
        }
        // output file chance
        if(voutput)
            dowrite(buff[buff_show_index],voutput);  // was +1
        ++count;
    }
    
    if(fout)
    {
        fclose(fout);
        fout = 0;
    }
    if(writer)
    {
        cvReleaseVideoWriter(writer);
        writer = 0;
    }

}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

