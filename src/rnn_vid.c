#include "network.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
image get_image_from_stream(CvCapture *cap);
image ipl_to_image(IplImage* src);

void reconstruct_picture(network net, float *features, image recon, image update, float rate, float momentum, float lambda, int smooth_size, int iters);


typedef struct {
    float *x;
    float *y;
} float_pair;

float_pair get_rnn_vid_data(network net, char **files, int n, int batch, int steps)
{
    int b;
    assert(net.batch == steps + 1);
    image out_im = get_network_image(net);
    int output_size = out_im.w*out_im.h*out_im.c;
    printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
    float *feats = calloc(net.batch*batch*output_size, sizeof(float));
    for(b = 0; b < batch; ++b){
        int input_size = net.w*net.h*net.c;
        float *input = calloc(input_size*net.batch, sizeof(float));
        char *filename = files[rand()%n];
        CvCapture *cap = cvCaptureFromFile(filename);
        int frames = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
        int index = rand() % (frames - steps - 2);
        if (frames < (steps + 4)){
            --b;
            free(input);
            continue;
        }

        printf("frames: %d, index: %d\n", frames, index);
        cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, index);

        int i;
        for(i = 0; i < net.batch; ++i){
            IplImage* src = cvQueryFrame(cap);
            image im = ipl_to_image(src);
            rgbgr_image(im);
            image re = resize_image(im, net.w, net.h);
            //show_image(re, "loaded");
            //cvWaitKey(10);
            memcpy(input + i*input_size, re.data, input_size*sizeof(float));
            free_image(im);
            free_image(re);
        }
        float *output = network_predict(net, input);

        free(input);

        for(i = 0; i < net.batch; ++i){
            memcpy(feats + (b + i*batch)*output_size, output + i*output_size, output_size*sizeof(float));
        }

        cvReleaseCapture(&cap);
    }

    //printf("%d %d %d\n", out_im.w, out_im.h, out_im.c);
    float_pair p = {0};
    p.x = feats;
    p.y = feats + output_size*batch; //+ out_im.w*out_im.h*out_im.c;

    return p;
}


void train_vid_rnn(char *cfgfile, char *weightfile)
{
    char *train_videos = "data/vid/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;

    list *plist = get_paths(train_videos);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);
    clock_t time;
    int steps = net.time_steps;
    int batch = net.batch / net.time_steps;

    network extractor = parse_network_cfg("cfg/extractor.cfg");
    load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        float_pair p = get_rnn_vid_data(extractor, paths, N, batch, steps);

        float loss = train_network_datum(net, p.x, p.y) / (net.batch);


        free(p.x);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time));
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%10==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


image save_reconstruction(network net, image *init, float *feat, char *name, int i)
{
    image recon;
    if (init) {
        recon = copy_image(*init);
    } else {
        recon = make_random_image(net.w, net.h, 3);
    }

    image update = make_image(net.w, net.h, 3);
    reconstruct_picture(net, feat, recon, update, .01, .9, .1, 2, 50);
    char buff[256];
    sprintf(buff, "%s%d", name, i);
    save_image(recon, buff);
    free_image(update);
    return recon;
}

void generate_vid_rnn(char *cfgfile, char *weightfile)
{
    network extractor = parse_network_cfg("cfg/extractor.recon.cfg");
    load_weights(&extractor, "/home/pjreddie/trained/yolo-coco.conv");

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&extractor, 1);
    set_batch_network(&net, 1);

    int i;
    CvCapture *cap = cvCaptureFromFile("/extra/vid/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00007030.mp4");
    float *feat;
    float *next;
	next = NULL;
    image last;
    for(i = 0; i < 25; ++i){
        image im = get_image_from_stream(cap);
        image re = resize_image(im, extractor.w, extractor.h);
        feat = network_predict(extractor, re.data);
        if(i > 0){
            printf("%f %f\n", mean_array(feat, 14*14*512), variance_array(feat, 14*14*512));
            printf("%f %f\n", mean_array(next, 14*14*512), variance_array(next, 14*14*512));
            printf("%f\n", mse_array(feat, 14*14*512));
            axpy_cpu(14*14*512, -1, feat, 1, next, 1);
            printf("%f\n", mse_array(next, 14*14*512));
        }
        next = network_predict(net, feat);

        free_image(im);

        free_image(save_reconstruction(extractor, 0, feat, "feat", i));
        free_image(save_reconstruction(extractor, 0, next, "next", i));
        if (i==24) last = copy_image(re);
        free_image(re);
    }
    for(i = 0; i < 30; ++i){
        next = network_predict(net, next);
        image new = save_reconstruction(extractor, &last, next, "new", i);
        free_image(last);
        last = new;
    }
}

void run_vid_rnn(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    //char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_vid_rnn(cfg, weights);
    else if(0==strcmp(argv[2], "generate")) generate_vid_rnn(cfg, weights);
}
#else
void run_vid_rnn(int argc, char **argv){}
#endif

