#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"

#ifdef OPENCV
#include <opencv2/highgui/highgui_c.h>
#endif

void train_cifar(char *cfgfile, char *weightfile)
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

    char* backup_directory = "backup/";
    int classes = 10;
    int N = 50000;

    char **labels = get_labels("data/cifar/labels.txt");
    int epoch = (*net.seen)/N;
    data train = load_all_cifar10();
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        clock_t time=clock();

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free(base);
    free_data(train);
}

void train_cifar_distill(char *cfgfile, char *weightfile)
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

    char* backup_directory = "backup/";
    int classes = 10;
    int N = 50000;

    char **labels = get_labels("data/cifar/labels.txt");
    int epoch = (*net.seen)/N;

    data train = load_all_cifar10();
    matrix soft = csv_to_matrix("results/ensemble.csv");

    float weight = .9;
    scale_matrix(soft, weight);
    scale_matrix(train.y, 1. - weight);
    matrix_add_matrix(soft, train.y);

    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        clock_t time=clock();

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free(base);
    free_data(train);
}

void test_cifar_multi(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(0));

    float avg_acc = 0;
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);

        float pred[10] = {0};

        float *p = network_predict(net, im.data);
        axpy_cpu(10, 1, p, 1, pred, 1);
        flip_image(im);
        p = network_predict(net, im.data);
        axpy_cpu(10, 1, p, 1, pred, 1);

        int index = max_index(pred, 10);
        int class_id = max_index(test.y.vals[i], 10);
        if(index == class_id) avg_acc += 1;
        free_image(im);
        printf("%4d: %.2f%%\n", i, 100.*avg_acc/(i+1));
    }
}

void test_cifar(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    clock_t time;
    float avg_acc = 0;
    float avg_top5 = 0;
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    time=clock();

    float *acc = network_accuracies(net, test, 2);
    avg_acc += acc[0];
    avg_top5 += acc[1];
    printf("top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock()-time), test.X.rows);
    free_data(test);
}

void extract_cifar()
{
char *labels[] = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
    int i;
    data train = load_all_cifar10();
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");
    for(i = 0; i < train.X.rows; ++i){
        image im = float_to_image(32, 32, 3, train.X.vals[i]);
        int class_id = max_index(train.y.vals[i], 10);
        char buff[256];
        sprintf(buff, "data/cifar/train/%d_%s",i,labels[class_id]);
        save_image_png(im, buff);
    }
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        int class_id = max_index(test.y.vals[i], 10);
        char buff[256];
        sprintf(buff, "data/cifar/test/%d_%s",i,labels[class_id]);
        save_image_png(im, buff);
    }
}

void test_cifar_csv(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    matrix pred = network_predict_data(net, test);

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        flip_image(im);
    }
    matrix pred2 = network_predict_data(net, test);
    scale_matrix(pred, .5);
    scale_matrix(pred2, .5);
    matrix_add_matrix(pred2, pred);

    matrix_to_csv(pred);
    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
}

void test_cifar_csvtrain(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    data test = load_all_cifar10();

    matrix pred = network_predict_data(net, test);

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        flip_image(im);
    }
    matrix pred2 = network_predict_data(net, test);
    scale_matrix(pred, .5);
    scale_matrix(pred2, .5);
    matrix_add_matrix(pred2, pred);

    matrix_to_csv(pred);
    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
}

void eval_cifar_csv()
{
    data test = load_cifar10_data("data/cifar/cifar-10-batches-bin/test_batch.bin");

    matrix pred = csv_to_matrix("results/combined.csv");
    fprintf(stderr, "%d %d\n", pred.rows, pred.cols);

    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
    free_matrix(pred);
}


void run_cifar(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_cifar(cfg, weights);
    else if(0==strcmp(argv[2], "extract")) extract_cifar();
    else if(0==strcmp(argv[2], "distill")) train_cifar_distill(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_cifar(cfg, weights);
    else if(0==strcmp(argv[2], "multi")) test_cifar_multi(cfg, weights);
    else if(0==strcmp(argv[2], "csv")) test_cifar_csv(cfg, weights);
    else if(0==strcmp(argv[2], "csvtrain")) test_cifar_csvtrain(cfg, weights);
    else if(0==strcmp(argv[2], "eval")) eval_cifar_csv();
}
