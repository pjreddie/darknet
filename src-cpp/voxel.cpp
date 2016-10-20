#include "network.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
image get_image_from_stream(CvCapture *cap);
#endif

void extract_voxel(char *lfile, char *rfile, char *prefix)
{
#ifdef OPENCV
    int w = 1920;
    int h = 1080;
    int shift = 0;
    int count = 0;
    CvCapture *lcap = cvCaptureFromFile(lfile);
    CvCapture *rcap = cvCaptureFromFile(rfile);
    while(1){
        image l = get_image_from_stream(lcap);
        image r = get_image_from_stream(rcap);
        if(!l.w || !r.w) break;
        if(count%100 == 0) {
            shift = best_3d_shift_r(l, r, -l.h/100, l.h/100);
            printf("%d\n", shift);
        }
        image ls = crop_image(l, (l.w - w)/2, (l.h - h)/2, w, h);
        image rs = crop_image(r, 105 + (r.w - w)/2, (r.h - h)/2 + shift, w, h);
        char buff[256];
        sprintf(buff, "%s_%05d_l", prefix, count);
        save_image(ls, buff);
        sprintf(buff, "%s_%05d_r", prefix, count);
        save_image(rs, buff);
        free_image(l);
        free_image(r);
        free_image(ls);
        free_image(rs);
        ++count;
    }

#else
    printf("need OpenCV for extraction\n");
#endif
}

void train_voxel(char *cfgfile, char *weightfile)
{
    char *train_images = "/data/imagenet/imagenet1k.train.list";
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
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.scale = 4;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = SUPER_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void test_voxel(char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        resize_network(&net, im.w, im.h);
        printf("%d %d\n", im.w, im.h);

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image(net);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        save_image(out, "out");

        free_image(im);
        if (filename) break;
    }
}


void run_voxel(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")) train_voxel(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_voxel(cfg, weights, filename);
    else if(0==strcmp(argv[2], "extract")) extract_voxel(argv[3], argv[4], argv[5]);
    /*
       else if(0==strcmp(argv[2], "valid")) validate_voxel(cfg, weights);
     */
}
