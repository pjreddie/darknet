#include "network.h"
#include "utils.h"
#include "parser.h"

char *dice_labels[] = {"face1","face2","face3","face4","face5","face6"};

void train_dice(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char* backup_directory = "backup/";
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    int i = *net.seen/imgs;
    char **labels = dice_labels;
    list *plist = get_paths("data/dice/dice.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_old(paths, imgs, plist->size, labels, 6, net.w, net.h);
        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %" PRIu64 " images\n", i, loss, avg_loss, sec(clock()-time), *net.seen);
        free_data(train);
        if((i % 100) == 0) net.learning_rate *= .1;
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, i);
            save_weights(net, buff);
        }
    }
}

void validate_dice(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    char **labels = dice_labels;
    list *plist = get_paths("data/dice/dice.val.list");

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    data val = load_data_old(paths, m, 0, labels, 6, net.w, net.h);
    float *acc = network_accuracies(net, val, 2);
    printf("Validation Accuracy: %f, %d images\n", acc[0], m);
    free_data(val);
}

void test_dice(char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    int i = 0;
    char **names = dice_labels;
    char buff[256];
    char *input = buff;
    int indexes[6];
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
        image im = load_image_color(input, net.w, net.h);
        float *X = im.data;
        float *predictions = network_predict(net, X);
        top_predictions(net, 6, indexes);
        for(i = 0; i < 6; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}

void run_dice(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_dice(cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_dice(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_dice(cfg, weights);
}
