#include "network.h"
#include "utils.h"
#include "parser.h"

void train_imagenet(char *cfgfile, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    //net.seen=0;
    int imgs = 1024;
    int i = net.seen/imgs;
    char **labels = get_labels("data/inet.labels.list");
    list *plist = get_paths("/data/imagenet/cls.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;
    load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 256, 256, &buffer);
    while(1){
        ++i;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        /*
        image im = float_to_image(256, 256, 3, train.X.vals[114]);
        show_image(im, "training");
        cvWaitKey(0);
        */

        load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 256, 256, &buffer);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), net.seen);
        free_data(train);
        if((i % 30000) == 0) net.learning_rate *= .1;
        //if(i%100 == 0 && net.learning_rate > .00001) net.learning_rate *= .97;
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}

void validate_imagenet(char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    char **labels = get_labels("data/inet.labels.list");
    list *plist = get_paths("data/inet.val.list");

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_top5 = 0;
    int splits = 50;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;
    pthread_t load_thread = load_data_thread(paths, num, 0, labels, 1000, 256, 256, &buffer);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits) load_thread = load_data_thread(part, num, 0, labels, 1000, 256, 256, &buffer);
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val);
        avg_acc += acc[0];
        avg_top5 += acc[1];
        printf("%d: top1: %f, top5: %f, %lf seconds, %d images\n", i, avg_acc/i, avg_top5/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void test_imagenet(char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    int i = 0;
    char **names = get_labels("data/shortnames.txt");
    clock_t time;
    char input[256];
    int indexes[10];
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            fgets(input, 256, stdin);
            strtok(input, "\n");
        }
        image im = load_image_color(input, 256, 256);
        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);
        top_predictions(net, 10, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < 10; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}

void run_imagenet(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_imagenet(cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_imagenet(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_imagenet(cfg, weights);
}

/*
   void train_imagenet_distributed(char *address)
   {
   float avg_loss = 1;
   srand(time(0));
   network net = parse_network_cfg("cfg/net.cfg");
   set_learning_network(&net, 0, 1, 0);
   printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
   int imgs = net.batch;
   int i = 0;
   char **labels = get_labels("/home/pjreddie/data/imagenet/cls.labels.list");
   list *plist = get_paths("/data/imagenet/cls.train.list");
   char **paths = (char **)list_to_array(plist);
   printf("%d\n", plist->size);
   clock_t time;
   data train, buffer;
   pthread_t load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 224, 224, &buffer);
   while(1){
   i += 1;

   time=clock();
   client_update(net, address);
   printf("Updated: %lf seconds\n", sec(clock()-time));

   time=clock();
   pthread_join(load_thread, 0);
   train = buffer;
   normalize_data_rows(train);
   load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 224, 224, &buffer);
   printf("Loaded: %lf seconds\n", sec(clock()-time));
   time=clock();

   float loss = train_network(net, train);
   avg_loss = avg_loss*.9 + loss*.1;
   printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs);
   free_data(train);
   }
   }
 */

