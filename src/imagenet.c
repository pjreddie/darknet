#include "network.h"
#include "utils.h"
#include "parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

void train_imagenet(char *cfgfile, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "/home/pjreddie/backup/";
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    char **labels = get_labels("data/inet.labels.list");
    list *plist = get_paths("data/inet.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = 1000;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    load_thread = load_data_in_thread(args);
    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        load_thread = load_data_in_thread(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        free_data(train);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(*net.seen%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    pthread_join(load_thread, 0);
    free_data(buffer);
    free_network(net);
    free_ptrs((void**)labels, 1000);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
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
    //list *plist = get_paths("data/inet.suppress.list");
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

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = 1000;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val, 5);
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
    int indexes[10];
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

