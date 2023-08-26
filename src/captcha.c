#include "network.h"
#include "utils.h"
#include "parser.h"

void fix_data_captcha(data d, int mask)
{
    matrix labels = d.y;
    int i, j;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; j += 2){
            if (mask){
                if(!labels.vals[i][j]){
                    labels.vals[i][j] = SECRET_NUM;
                    labels.vals[i][j+1] = SECRET_NUM;
                }else if(labels.vals[i][j+1]){
                    labels.vals[i][j] = 0;
                }
            } else{
                if (labels.vals[i][j]) {
                    labels.vals[i][j+1] = 0;
                } else {
                    labels.vals[i][j+1] = 1;
                }
            }
        }
    }
}

void train_captcha(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    int i = *net.seen/imgs;
    int solved = 1;
    list *plist;
    char** labels = get_labels("data/captcha/reimgs.labels.list");
    if (solved){
        plist = get_paths("data/captcha/reimgs.solved.list");
    }else{
        plist = get_paths("data/captcha/reimgs.raw.list");
    }
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = 26;
    args.n = imgs;
    args.m = plist->size;
    args.labels = labels;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;

    load_thread = load_data_in_thread(args);
    while(1){
        ++i;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        fix_data_captcha(train, solved);

        /*
           image im = float_to_image(256, 256, 3, train.X.vals[114]);
           show_image(im, "training");
           cvWaitKey(0);
         */

        load_thread = load_data_in_thread(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %" PRIu64 " images\n", i, loss, avg_loss, sec(clock()-time), *net.seen);
        free_data(train);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "imagenet_backup/%s_%d.weights", base, i);
            save_weights(net, buff);
        }
    }
}

void test_captcha(char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    int i = 0;
    char** names = get_labels("data/captcha/reimgs.labels.list");
    char buff[256];
    char *input = buff;
    int indexes[26];
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            //printf("Enter Image Path: ");
            //fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, net.w, net.h);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        top_predictions(net, 26, indexes);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < 26; ++i){
            int index = indexes[i];
            if(i != 0) printf(", ");
            printf("%s %f", names[index], predictions[index]);
        }
        printf("\n");
        fflush(stdout);
        free_image(im);
        if (filename) break;
    }
}

void valid_captcha(char *cfgfile, char *weightfile, char *filename)
{
    char** labels = get_labels("data/captcha/reimgs.labels.list");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    list* plist = get_paths("data/captcha/reimgs.fg.list");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    int outputs = net.outputs;

    set_batch_network(&net, 1);
    srand(2222222);
    int i, j;
    for(i = 0; i < N; ++i){
        if (i%100 == 0) fprintf(stderr, "%d\n", i);
        image im = load_image_color(paths[i], net.w, net.h);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        int truth = -1;
        for(j = 0; j < 13; ++j){
            if (strstr(paths[i], labels[j])) truth = j;
        }
        if (truth == -1){
            fprintf(stderr, "bad: %s\n", paths[i]);
            return;
        }
        printf("%d, ", truth);
        for(j = 0; j < outputs; ++j){
            if (j != 0) printf(", ");
            printf("%f", predictions[j]);
        }
        printf("\n");
        fflush(stdout);
        free_image(im);
        if (filename) break;
    }
}

/*
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
    */
void run_captcha(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_captcha(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_captcha(cfg, weights, filename);
    else if(0==strcmp(argv[2], "valid")) valid_captcha(cfg, weights, filename);
    //if(0==strcmp(argv[2], "test")) test_captcha(cfg, weights);
    //else if(0==strcmp(argv[2], "encode")) encode_captcha(cfg, weights);
    //else if(0==strcmp(argv[2], "decode")) decode_captcha(cfg, weights);
    //else if(0==strcmp(argv[2], "valid")) validate_captcha(cfg, weights);
}
