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
    int imgs = 256;
    int i = net.seen/imgs;
    list *plist = get_paths("data/train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_writing(paths, imgs, plist->size, 256, 256, 4);
        float loss = train_network(net, train);
        #ifdef GPU
        float *out = get_network_output_gpu(net);
        #else
        float *out = get_network_output(net);
        #endif
//        image pred = float_to_image(32, 32, 1, out);
//        print_image(pred);

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
        if(i%250==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights", base, i);
            save_weights(net, buff);
        }
    }
}

void test_writing(char *cfgfile, char *weightfile, char *outfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char filename[256];

    fgets(filename, 256, stdin);
    strtok(filename, "\n");
    image im = load_image_color(filename, 0, 0);
    //image im = load_image_color("/home/pjreddie/darknet/data/figs/C02-1001-Figure-1.png", 0, 0);
    image sized = resize_image(im, 256, 256);
    printf("%d %d %d\n", im.h, im.w, im.c);
    float *X = sized.data;
    time=clock();
    float *predictions = network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
    image pred = get_network_image(net);

    if (outfile) {
        printf("Save image as %s.png (shape: %d %d)\n", outfile, pred.w, pred.h);
        save_image(pred, outfile);
    } else {
        show_image(pred, "prediction");
        cvWaitKey(0);
    }

	free_image(im);
	free_image(sized);
}

void run_writing(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *outfile = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")) train_writing(cfg, weights);
    if(0==strcmp(argv[2], "test")) test_writing(cfg, weights, outfile);
}

