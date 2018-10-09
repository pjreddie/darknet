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
#include <semaphore.h>
#include <stdbool.h>

#ifdef TS

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#define EVENT_DEVICE "/dev/input/event4"
#define EVENT_TYPE EV_ABS
#define EVENT_CODE_X ABS_X
#define EVENT_CODE_Y ABS_Y

// ubuntu is force to be displayed on full screen
// hide taskbar in full screen otherwise it will not work perfectly
// Touch screen option
static double X_MIN = 0.;
static double Y_MIN = 0.;
static double X_MAX = 1024.;
static double Y_MAX = 600.;
// Camera option
static double CAM_W = 640.;
static double CAM_H = 480.;
static double RCOORD[2] = {-1., -1.};

#endif

static int autotrack_flag = 1; //0: track target defined by touch screen; 1: track target defined by detection;
static int detect_flag = 0;    //0: target not found; 1: predefined target found by detection;
static int lost_flag = 0;
static int imagewriting_flag = 0;

#define DEMO 1

#ifdef OPENCV    
#include <cv.h>
#include <highgui.h>
#include <opencv/highgui.h>
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff[3]; //0-fetch;1-display;2-detect;
static image buff_letter[3];
static int buff_index = 0;
static void * cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for (j = 0; j < demo_frame; ++j)
    {
        axpy_cpu(demo_total, 1. / demo_frame, predictions[j], 1, avg, 1);
    }
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

//detect image buff[(buff_index+2)%3]
void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n - 1];
    float *X = buff_letter[(buff_index + 2) % 3].data;
    network_predict(net, X);

    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);

    if (nms > 0)
        do_nms_obj(dets, nboxes, l.classes, nms);
    //clear the screen to show fixed;
    /*    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n"); */
    image display = buff[(buff_index + 2) % 3];

    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);

    // save tracking target parameters in target.txt
    if (autotrack_flag == 0)
    {
#ifdef TS
        if (RCOORD[0] >= 0. && RCOORD[1] >= 0.)
        {
            double trcoord[2] = {RCOORD[0], RCOORD[1]};
            save_TS_target(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, trcoord);
            RCOORD[0] = trcoord[0];
            RCOORD[1] = trcoord[1];
        }
#endif
    }
    else
    {
        save_autotrack_target(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    }

    save_image(display, "yolo"); //save image as png file.
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1) % demo_frame;
    running = 0;
    return 0;
}

void *track_in_thread(void *ptr)
{
/*<<<<<<< HEAD
    char buf[10];
    pthread_t detect_thread;
    sleep(5);
    while (1)
    {
        // Read flags from txt;
        FILE *f = fopen("tracker.txt", "r");
        if (f == NULL)
        {
            printf("ERROR opening tracker.txt.\n");
        }
        while (fgets(buf, 10, f) != NULL)
            //       printf("%s", buf);
            detect_flag = buf[0] - '0';
        lost_flag = buf[2] - '0';
        imagewriting_flag = buf[4] - '0';
        fclose(f);
        //printf("flag:%d,%d,%d\n",detect_flag, lost_flag, imagewriting_flag);
        if (detect_flag == 0 && lost_flag == 1 && imagewriting_flag == 0)
        {
            //printf("flag1\n");
            cap = cvCaptureFromFile("tracker.png");
            if (!cap)
                error("Couldn't read tracker.png.\n");
            buff[0] = get_image_from_stream(cap);
            buff[1] = copy_image(buff[0]);
            buff[2] = copy_image(buff[0]);
            buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
            buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
            buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
            buff_index = (buff_index + 1) % 3; //go to next image buff
            if (pthread_create(&detect_thread, 0, detect_in_thread, 0))
                error("Thread creation failed");
            pthread_join(detect_thread, 0);

            cvReleaseCapture(&cap);
        }

        //sleep(0.1);
    }
=======*/
    free_image(buff[buff_index]);
    buff[buff_index] = get_image_from_stream(cap);
    if(buff[buff_index].data == 0) {
        demo_done = 1;
        return 0;
    }
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    return 0;
}

//display image of buff[(buff_index+1)%3];
void *display_in_thread(void *ptr)
{
    int c = show_image(buff[(buff_index + 1)%3], "Demo", 1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    }
    else if (c == 116)
    { // press button "t"
        demo_done = 1;
    }
    else if (c == 82)
    {
        demo_thresh += .02;
    }
    else if (c == 84)
    {
        demo_thresh -= .02;
        if (demo_thresh <= .02)
            demo_thresh = .02;
    }
    else if (c == 83)
    {
        demo_hier += .02;
    }
    else if (c == 81)
    {
        demo_hier -= .02;
        if (demo_hier <= .0)
            demo_hier = .0;
    }
    return 0;
}
//fetch image to buff[buff_index];
void *fetch_in_thread(void *ptr)
{
    free_image(buff[buff_index]);
    buff[buff_index] = get_image_from_stream(cap);
    if(buff[buff_index].data == 0) {
        demo_done = 1;
        return 0;
    }
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    return 0;
}

void *display_loop(void *ptr)
{
    while (1)
    {
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while (1)
    {
        detect_in_thread(0);
    }
}

void save_autotrack_target(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
    int i, j, i_x, i_y, i_w, i_h;
    bool flag = false;
    char labelref[4096] = {0};

    for (i = 0; i < num; ++i)
    {
        char labelstr[4096] = {0};
        int class = -1;
        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j] > thresh)
            {
                if (class < 0)
                {
                    strcat(labelstr, names[j]);
                    class = j;
                }
                else
                {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                //printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
        if (class >= 0)
        {
            box b = dets[i].bbox;
            printf("labelstr:%s\n", labelstr);

            if (!strcmp(labelstr, "person"))
            {
                i_x = (b.x - b.w / 2.) * im.w;
                i_y = (b.y - b.h / 2.) * im.h;
                i_w = b.w * im.w;
                i_h = b.h * im.h;
                strcpy(labelref, labelstr);
                flag = true;
                detect_flag = 1;
                demo_done = 1;
                break;
            }
        }
    }
    if (flag != true)
    {
        strcat(labelref, "N");
        i_x = 0;
        i_y = 0;
        i_w = 0;
        i_h = 0;
        detect_flag = 0;
    }
    printf("demo.c - Tracking target: %s,%d,%d,%d,%d,%d\n", labelref, i_x, i_y, i_w, i_h, detect_flag);
    FILE *f = fopen("yolo.txt", "w");
    if (f == NULL)
    {
        printf("ERROR opening yolo.txt.\n");
    }
    fprintf(f, "%s,%d,%d,%d,%d,%d\n", labelref, i_x, i_y, i_w, i_h, detect_flag); //format: label,x,y,w,h
    fclose(f);
}

#ifdef TS
void save_TS_target(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, double *coord)
{
    int i, j, i_x, i_y, i_w, i_h;
    float distance = 1000000000.0, tdistance = 0.0;
    char labelref[4096] = {0};

    for (i = 0; i < num; ++i)
    {
        char labelstr[4096] = {0};
        int class = -1;
        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j] > thresh)
            {
                if (class < 0)
                {
                    strcat(labelstr, names[j]);
                    class = j;
                }
                else
                {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                //printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
        if (class >= 0)
        {

            box b = dets[i].bbox;

            tdistance = sqrt(pow(coord[0] - b.x, 2.) + pow(coord[1] - b.y, 2.));
            //get the box with the shortest distance;
            if (distance > tdistance)
            {
                i_x = (b.x - b.w / 2.) * im.w;
                i_y = (b.y - b.h / 2.) * im.h;
                i_w = b.w * im.w;
                i_h = b.h * im.h;
                distance = tdistance;
                strcpy(labelref, labelstr);
            }
        }
    }
    printf("demo.c - Selected point: %f %f \n", coord[0] * im.w, coord[1] * im.h); // touch coordinate
    printf("demo.c - Tracking target: %s,%d,%d,%d,%d\n", labelref, i_x, i_y, i_w, i_h);
    FILE *f = fopen("yolo.txt", "w");
    if (f == NULL)
    {
        printf("ERROR opening capture.txt to save the box that needed to follow.\n");
    }
    fprintf(f, "%s,%d,%d,%d,%d\n", labelref, i_x, i_y, i_w, i_h); //format: label,x,y,w,h
    fclose(f);
}
// read coordiantes X,Y from TS
// into RCOORD
// (New function define by Raphael)
int change_coord()
{
    //double ratio=min((xTSmax-xTS0)/xcam,(yTSmax-yTS0)/ycam)
    double ratio = (Y_MAX - Y_MIN) / CAM_H;
    double ts_w = (X_MAX - X_MIN);
    double ts_h = (Y_MAX - Y_MIN);
    double new_w = (CAM_W)*ratio;
    double new_h = (CAM_H)*ratio;

    // printf("ratio %f new w %f new h %f \n",ratio,new_w,new_h);
    double left = X_MIN + (ts_w - new_w) / 2.0;
    double right = X_MAX - (ts_w - new_w) / 2.0;
    double top = Y_MIN + (ts_h - new_h) / 2.0;
    double bot = Y_MAX - (ts_h - new_h) / 2.0;
    printf("\nTS corners: left %f right %f top %f bot %f \n", left, right, top, bot);

    if (left < RCOORD[0] && RCOORD[0] < right && top < RCOORD[1] && RCOORD[0] < bot)
    {
        RCOORD[0] = (RCOORD[0] - left) / ts_w;
        RCOORD[1] = (RCOORD[1] - top) / ts_h;
        return 1;
    }
    else
    {
        printf("\nPixel selected is not in the BBOX. Try again\n");
        RCOORD[0] = -1.;
        RCOORD[1] = -1.;
        return 0;
    }
}
// Detect if the TS is touch
// IS the pixel selected in the frame?
// Add pixel into RCOORD
// (New function define by Raphael)
int input_TS(int fileId)
{
    fd_set set;
    struct timeval timeout;

    /* Initialize the file descriptor set. */
    FD_ZERO(&set);
    FD_SET(fileId, &set);

    /* Initialize the timeout data structure. */
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;

    /* select returns 0 if timeout, 1 if input available, -1 if error. */
    if (select(FD_SETSIZE, &set, NULL, NULL, &timeout))
    {
        const size_t ev_size = sizeof(struct input_event);
        ssize_t size;
        struct input_event ev;
        size = read(fileId, &ev, ev_size);
        if (size < ev_size)
        {
            error("Error size when reading\n");
        }

        if (ev.type == EVENT_TYPE && (ev.code == EVENT_CODE_X || ev.code == EVENT_CODE_Y))
        {

            RCOORD[0] = ev.value;
            printf("Selected %s = %f\n", "X", RCOORD[0]);
            size = read(fileId, &ev, ev_size);

            RCOORD[1] = ev.value;
            printf("Selected %s = %f\n", "Y", RCOORD[1]);

            return change_coord();
        }
    }
    return 0;
}
#endif

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float *));
    for (i = 0; i < demo_frame; ++i)
    {
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if (filename)
    {
        printf("video file: %s\n", filename);
        cap = open_video_stream(filename, 0, 0, 0, 0);
    }else{
        cap = open_video_stream(0, cam_index, w, h, frames);
    }

    if (!cap)
        error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    int count = 0;
    if(!prefix){
        make_window("Demo", 1352, 1013, fullscreen);
    }

    demo_time = what_time_is_it_now();

    while (!demo_done)
    {
        buff_index = (buff_index + 1) % 3;
        if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0))
            error("Thread creation failed");
        if (pthread_create(&detect_thread, 0, detect_in_thread, 0))
            error("Thread creation failed");
        if (!prefix)
        {
            fps = 1. / (what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }
        else
        {
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1) % 3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
    cvDestroyWindow("Demo");
    cvReleaseCapture(&cap);
}
/*
#ifdef OPENTRACKER
void tracking(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    if (remove("tracker.txt") != 0)
        perror("Error deleting tracker.txt");

    if (remove("yolo.txt") != 0)
        perror("Error deleting yolo.txt");

    if (remove("tracker.png") != 0)
        perror("Error deleting tracker.png");

    if (remove("yolo.png") != 0)
        perror("Error deleting yolo.png");

    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;

    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;
    pthread_t track_thread;

    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float *));
    for (i = 0; i < demo_frame; ++i)
    {
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    cap = cvCaptureFromCAM(cam_index);
    if (w)
    {
        cvSetCaptureProperty(cap, 3, w);
    }
    if (h)
    {
        cvSetCaptureProperty(cap, 4, h);
    }
    if (frames)
    {
        cvSetCaptureProperty(cap, 5, frames);
    }
    if (!cap)
        error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h); //squeez the image to a letterbox
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    //ipl = cvCreateImage(cvSize(buff[0].w, buff[0].h), IPL_DEPTH_8U, buff[0].c);

    cvNamedWindow("Demo", CV_WINDOW_NORMAL);
    if (fullscreen)
    {
        cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    }
    else
    {
        cvMoveWindow("Demo", 0, 0);
        cvResizeWindow("Demo", 1352, 1013);
    }

    while (!demo_done)
    {
        buff_index = (buff_index + 1) % 3; //go to next image buff
        if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0))
            error("Thread creation failed");
        if (pthread_create(&detect_thread, 0, detect_in_thread, 0))
            error("Thread creation failed");

        display_in_thread(0); //show image;

        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
    }
    cvDestroyWindow("Demo");
    cvReleaseCapture(&cap);

    if (pthread_create(&track_thread, 0, track_in_thread, 0))
        error("Thread creation failed");

    trackersdarknet();

    printf("End of tracking.\n");
}
#endif

#ifdef TS
// General loop detect TS and detect Target
// (New function define by Raphael)
void demo_TS(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("\nDemo track\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float *));
    for (i = 0; i < demo_frame; ++i)
    {
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    cap = cvCaptureFromCAM(cam_index);

    if (w)
    {
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        printf("width %d\n", w);
    }
    if (h)
    {
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        printf("height %d\n", h);
    }
    if (frames)
    {
        cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        printf("Nb Frames %d\n", frames);
    }
    // Start modification Raphel
    //    }
    // End modification Raphael
    if (!cap)
        error("Couldn't connect to webcam.\n");
    // Start Modification raphael
    CAM_H = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT);
    CAM_W = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH);

    printf("cam height %f\n", CAM_H);
    printf("cam width  %f\n", CAM_W);
    // End modification raphael

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    ipl = cvCreateImage(cvSize(buff[0].w, buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if (!prefix)
    {
        cvNamedWindow("Demo", CV_WINDOW_NORMAL);
        cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    }

    // Start modification raphael
    // Check if root
    if ((getuid()) != 0)
    {
        error("You are not root! This may not work...\n");
    }

    // Open Device TS
    int fd;
    if (filename)
    {
        printf("\nTouch screen read from %s\n", filename);
        fd = open(filename, O_RDONLY | O_NONBLOCK);
    }
    else
    {
        printf("\nTouch screen read from %s\n", EVENT_DEVICE);
        fd = open(EVENT_DEVICE, O_RDONLY | O_NONBLOCK);
    }

    if (fd == -1)
    {
        error("Event is not a vaild device\n");
    }

    // Print Device Name
    char name[256] = "Unknown";
    ioctl(fd, EVIOCGNAME(sizeof(name)), name);
    ioctl(fd, EVIOCGPROP(sizeof(name)), name);

    printf("Reading from:\n");
    printf("Device file = %s\n", filename != NULL ? filename : EVENT_DEVICE);
    printf("Device name = %s\n", name);
    printf("Taille X = %d\n", ABS_X);
    printf("Taille Y = %d\n", ABS_Y);
    // End modification raphael

    demo_time = what_time_is_it_now();

    // Start modification raphael
    int ts = 0;
    while (!demo_done && !ts)
    {
        ts = input_TS(fd);
        // End modification Raphael
        buff_index = (buff_index + 1) % 3;
        if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0))
            error("Thread creation failed");
        if (pthread_create(&detect_thread, 0, detect_in_thread, 0))
            error("Thread creation failed");
        if (!prefix)
        {
            fps = 1. / (what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }
        else
        {
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1) % 3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
    // Start modification Raphael
    close(fd);
    cvDestroyWindow("Demo");
    cvReleaseCapture(&cap);
    // End modification Raphael
}
#endif
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
