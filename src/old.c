
void test_load()
{
    image dog = load_image("dog.jpg", 300, 400);
    show_image(dog, "Test Load");
    show_image_layers(dog, "Test Load");
}

void test_parser()
{
    network net = parse_network_cfg("cfg/trained_imagenet.cfg");
    save_network(net, "cfg/trained_imagenet_smaller.cfg");
}

void test_init(char *cfgfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    srand(2222222);
    int i = 0;
    char *filename = "data/test.jpg";

    image im = load_image_color(filename, 256, 256);
    //z_normalize_image(im);
    translate_image(im, -128);
    scale_image(im, 1/128.);
    float *X = im.data;
    forward_network(net, X, 0, 1);
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            image output = get_convolutional_image(layer);
            int size = output.h*output.w*output.c;
            float v = variance_array(layer.output, size);
            float m = mean_array(layer.output, size);
            printf("%d: Convolutional, mean: %f, variance %f\n", i, m, v);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            int size = layer.outputs;
            float v = variance_array(layer.output, size);
            float m = mean_array(layer.output, size);
            printf("%d: Connected, mean: %f, variance %f\n", i, m, v);
        }
    }
    free_image(im);
}
void test_dog(char *cfgfile)
{
    image im = load_image_color("data/dog.jpg", 256, 256);
    translate_image(im, -128);
    print_image(im);
    float *X = im.data;
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    network_predict(net, X);
    image crop = get_network_image_layer(net, 0);
    show_image(crop, "cropped");
    print_image(crop);
    show_image(im, "orig");
    float * inter = get_network_output(net);
    pm(1000, 1, inter);
    cvWaitKey(0);
}

void test_voc_segment(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    while(1){
        char filename[256];
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        image im = load_image_color(filename, 500, 500);
        //resize_network(net, im.h, im.w, im.c);
        translate_image(im, -128);
        scale_image(im, 1/128.);
        //float *predictions = network_predict(net, im.data);
        network_predict(net, im.data);
        free_image(im);
        image output = get_network_image_layer(net, net.n-2);
        show_image(output, "Segment Output");
        cvWaitKey(0);
    }
}
void test_visualize(char *filename)
{
    network net = parse_network_cfg(filename);
    visualize_network(net);
    cvWaitKey(0);
}

void test_cifar10(char *cfgfile)
{
    network net = parse_network_cfg(cfgfile);
    data test = load_cifar10_data("data/cifar10/test_batch.bin");
    clock_t start = clock(), end;
    float test_acc = network_accuracy_multi(net, test, 10);
    end = clock();
    printf("%f in %f Sec\n", test_acc, sec(end-start));
    //visualize_network(net);
    //cvWaitKey(0);
}

void train_cifar10(char *cfgfile)
{
    srand(555555);
    srand(time(0));
    network net = parse_network_cfg(cfgfile);
    data test = load_cifar10_data("data/cifar10/test_batch.bin");
    int count = 0;
    int iters = 50000/net.batch;
    data train = load_all_cifar10();
    while(++count <= 10000){
        clock_t time = clock();
        float loss = train_network_sgd(net, train, iters);

        if(count%10 == 0){
            float test_acc = network_accuracy(net, test);
            printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds\n", count, loss, test_acc,sec(clock()-time));
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/cifar10_%d.cfg", count);
            save_network(net, buff);
        }else{
            printf("%d: Loss: %f, Time: %lf seconds\n", count, loss, sec(clock()-time));
        }

    }
    free_data(train);
}

void compare_nist(char *p1,char *p2)
{
    srand(222222);
    network n1 = parse_network_cfg(p1);
    network n2 = parse_network_cfg(p2);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    normalize_data_rows(test);
    compare_networks(n1, n2, test);
}

void test_nist(char *path)
{
    srand(222222);
    network net = parse_network_cfg(path);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    normalize_data_rows(test);
    clock_t start = clock(), end;
    float test_acc = network_accuracy(net, test);
    end = clock();
    printf("Accuracy: %f, Time: %lf seconds\n", test_acc,(float)(end-start)/CLOCKS_PER_SEC);
}

void train_nist(char *cfgfile)
{
    srand(222222);
    // srand(time(0));
    data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    network net = parse_network_cfg(cfgfile);
    int count = 0;
    int iters = 6000/net.batch + 1;
    while(++count <= 100){
        clock_t start = clock(), end;
        normalize_data_rows(train);
        normalize_data_rows(test);
        float loss = train_network_sgd(net, train, iters);
        float test_acc = 0;
        if(count%1 == 0) test_acc = network_accuracy(net, test);
        end = clock();
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC);
    }
    free_data(train);
    free_data(test);
    char buff[256];
    sprintf(buff, "%s.trained", cfgfile);
    save_network(net, buff);
}

/*
   void train_nist_distributed(char *address)
   {
   srand(time(0));
   network net = parse_network_cfg("cfg/nist.client");
   data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
//data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
normalize_data_rows(train);
//normalize_data_rows(test);
int count = 0;
int iters = 50000/net.batch;
iters = 1000/net.batch + 1;
while(++count <= 2000){
clock_t start = clock(), end;
float loss = train_network_sgd(net, train, iters);
client_update(net, address);
end = clock();
//float test_acc = network_accuracy_gpu(net, test);
//float test_acc = 0;
printf("%d: Loss: %f, Time: %lf seconds\n", count, loss, (float)(end-start)/CLOCKS_PER_SEC);
}
}
 */

void test_ensemble()
{
    int i;
    srand(888888);
    data d = load_categorical_data_csv("mnist/mnist_train.csv", 0, 10);
    normalize_data_rows(d);
    data test = load_categorical_data_csv("mnist/mnist_test.csv", 0,10);
    normalize_data_rows(test);
    data train = d;
    //   data *split = split_data(d, 1, 10);
    //   data train = split[0];
    //   data test = split[1];
    matrix prediction = make_matrix(test.y.rows, test.y.cols);
    int n = 30;
    for(i = 0; i < n; ++i){
        int count = 0;
        float lr = .0005;
        float momentum = .9;
        float decay = .01;
        network net = parse_network_cfg("nist.cfg");
        while(++count <= 15){
            float acc = train_network_sgd(net, train, train.X.rows);
            printf("Training Accuracy: %lf Learning Rate: %f Momentum: %f Decay: %f\n", acc, lr, momentum, decay );
            lr /= 2; 
        }
        matrix partial = network_predict_data(net, test);
        float acc = matrix_topk_accuracy(test.y, partial,1);
        printf("Model Accuracy: %lf\n", acc);
        matrix_add_matrix(partial, prediction);
        acc = matrix_topk_accuracy(test.y, prediction,1);
        printf("Current Ensemble Accuracy: %lf\n", acc);
        free_matrix(partial);
    }
    float acc = matrix_topk_accuracy(test.y, prediction,1);
    printf("Full Ensemble Accuracy: %lf\n", acc);
}

void visualize_cat()
{
    network net = parse_network_cfg("cfg/voc_imagenet.cfg");
    image im = load_image_color("data/cat.png", 0, 0);
    printf("Processing %dx%d image\n", im.h, im.w);
    resize_network(net, im.h, im.w, im.c);
    forward_network(net, im.data, 0, 0);

    visualize_network(net);
    cvWaitKey(0);
}

void test_correct_nist()
{
    network net = parse_network_cfg("cfg/nist_conv.cfg");
    srand(222222);
    net = parse_network_cfg("cfg/nist_conv.cfg");
    data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    normalize_data_rows(train);
    normalize_data_rows(test);
    int count = 0;
    int iters = 1000/net.batch;

    while(++count <= 5){
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, iters);
        end = clock();
        float test_acc = network_accuracy(net, test);
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
    }
    save_network(net, "cfg/nist_gpu.cfg");

    gpu_index = -1;
    count = 0;
    srand(222222);
    net = parse_network_cfg("cfg/nist_conv.cfg");
    while(++count <= 5){
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, iters);
        end = clock();
        float test_acc = network_accuracy(net, test);
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
    }
    save_network(net, "cfg/nist_cpu.cfg");
}

void test_correct_alexnet()
{
    char **labels = get_labels("/home/pjreddie/data/imagenet/cls.labels.list");
    list *plist = get_paths("/data/imagenet/cls.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    int count = 0;
    network net;

    srand(222222);
    net = parse_network_cfg("cfg/net.cfg");
    int imgs = net.batch;

    count = 0;
    while(++count <= 5){
        time=clock();
        data train = load_data(paths, imgs, plist->size, labels, 1000, 256, 256);
        normalize_data_rows(train);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        printf("%d: %f, %lf seconds, %d images\n", count, loss, sec(clock()-time), imgs*net.batch);
        free_data(train);
    }

    gpu_index = -1;
    count = 0;
    srand(222222);
    net = parse_network_cfg("cfg/net.cfg");
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    while(++count <= 5){
        time=clock();
        data train = load_data(paths, imgs, plist->size, labels, 1000, 256,256);
        normalize_data_rows(train);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        printf("%d: %f, %lf seconds, %d images\n", count, loss, sec(clock()-time), imgs*net.batch);
        free_data(train);
    }
}

/*
   void run_server()
   {
   srand(time(0));
   network net = parse_network_cfg("cfg/net.cfg");
   set_batch_network(&net, 1);
   server_update(net);
   }

   void test_client()
   {
   network net = parse_network_cfg("cfg/alexnet.client");
   clock_t time=clock();
   client_update(net, "localhost");
   printf("1\n");
   client_update(net, "localhost");
   printf("2\n");
   client_update(net, "localhost");
   printf("3\n");
   printf("Transfered: %lf seconds\n", sec(clock()-time));
   }
 */
