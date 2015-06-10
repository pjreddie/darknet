#include "network.h"
#include "utils.h"
#include "parser.h"


void train_captcha(char *cfgfile, char *weightfile)
{
    float avg_loss = -1;
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    int i = net.seen/imgs;
    list *plist = get_paths("/data/captcha/train.auto5");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_captcha(paths, imgs, plist->size, 10, 200, 60);
        translate_data_rows(train, -128);
        scale_data_rows(train, 1./128);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), net.seen);
        free_data(train);
        if(i%10==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}

void decode_captcha(char *cfgfile, char *weightfile)
{
    setbuf(stdout, NULL);
    srand(time(0));
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    char filename[256];
    while(1){
        printf("Enter filename: ");
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        image im = load_image_color(filename, 300, 57);
        scale_image(im, 1./255.);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        image out  = float_to_image(300, 57, 1, predictions);
        show_image(out, "decoded");
        #ifdef OPENCV
        cvWaitKey(0);
        #endif
        free_image(im);
    }
}

void encode_captcha(char *cfgfile, char *weightfile)
{
    float avg_loss = -1;
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    int i = net.seen/imgs;
    list *plist = get_paths("/data/captcha/encode.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_captcha_encode(paths, imgs, plist->size, 300, 57);
        scale_data_rows(train, 1./255);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), net.seen);
        free_matrix(train.X);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}

void validate_captcha(char *cfgfile, char *weightfile)
{
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int numchars = 37;
    list *plist = get_paths("/data/captcha/solved.hard");
    char **paths = (char **)list_to_array(plist);
    int imgs = plist->size;
    data valid = load_data_captcha(paths, imgs, 0, 10, 200, 60);
    translate_data_rows(valid, -128);
    scale_data_rows(valid, 1./128);
    matrix pred = network_predict_data(net, valid);
    int i, k;
    int correct = 0;
    int total = 0;
    int accuracy = 0;
    for(i = 0; i < imgs; ++i){
        int allcorrect = 1;
        for(k = 0; k < 10; ++k){
            char truth = int_to_alphanum(max_index(valid.y.vals[i]+k*numchars, numchars));
            char prediction = int_to_alphanum(max_index(pred.vals[i]+k*numchars, numchars));
            if (truth != prediction) allcorrect=0;
            if (truth != '.' && truth == prediction) ++correct;
            if (truth != '.' || truth != prediction) ++total;
        }
        accuracy += allcorrect;
    }
    printf("Word Accuracy: %f, Char Accuracy %f\n", (float)accuracy/imgs, (float)correct/total);
    free_data(valid);
}

void test_captcha(char *cfgfile, char *weightfile)
{
    setbuf(stdout, NULL);
    srand(time(0));
    //char *base = basecfg(cfgfile);
    //printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    char filename[256];
    while(1){
        //printf("Enter filename: ");
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        image im = load_image_color(filename, 200, 60);
        translate_image(im, -128);
        scale_image(im, 1/128.);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        print_letters(predictions, 10);
        free_image(im);
    }
}
void run_captcha(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "test")) test_captcha(cfg, weights);
    else if(0==strcmp(argv[2], "train")) train_captcha(cfg, weights);
    else if(0==strcmp(argv[2], "encode")) encode_captcha(cfg, weights);
    else if(0==strcmp(argv[2], "decode")) decode_captcha(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_captcha(cfg, weights);
}

