#include "network.h"
#include "utils.h"
#include "parser.h"

void train_writing(char *cfgfile, char *weightfile)
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
    int imgs = 1024;
    int i = net.seen/imgs;
    list *plist = get_paths("figures.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_writing(paths, imgs, plist->size, 512, 512);
        float loss = train_network(net, train);
        #ifdef GPU
        float *out = get_network_output_gpu(net);
        #else
        float *out = get_network_output(net);
        #endif
        image pred = float_to_image(64, 64, 1, out);
        print_image(pred);

/*
        image im = float_to_image(256, 256, 3, train.X.vals[0]);
        image lab = float_to_image(64, 64, 1, train.y.vals[0]);
        image pred = float_to_image(64, 64, 1, out);
        show_image(im, "image");
        show_image(lab, "label");
        print_image(lab);
        show_image(pred, "pred");
        cvWaitKey(0);
        */

        net.seen += imgs;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), net.seen);
        free_data(train);
        if((i % 20000) == 0) net.learning_rate *= .1;
        //if(i%100 == 0 && net.learning_rate > .00001) net.learning_rate *= .97;
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}

void run_writing(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_writing(cfg, weights);
}

