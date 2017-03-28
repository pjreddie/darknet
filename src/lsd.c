#include "network.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

void train_lsd3(char *fcfg, char *fweight, char *gcfg, char *gweight, char *acfg, char *aweight, int clear)
{
#ifdef GPU
    //char *train_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    char *train_images = "/home/pjreddie/data/imagenet/imagenet1k.train.list";
    //char *style_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    char *style_images = "/home/pjreddie/zelda.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    network fnet = load_network(fcfg, fweight, clear);
    network gnet = load_network(gcfg, gweight, clear);
    network anet = load_network(acfg, aweight, clear);
    char *gbase = basecfg(gcfg);
    char *abase = basecfg(acfg);

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", gnet.learning_rate, gnet.momentum, gnet.decay);
    int imgs = gnet.batch*gnet.subdivisions;
    int i = *gnet.seen/imgs;
    data train, tbuffer;
    data style, sbuffer;


    list *slist = get_paths(style_images);
    char **spaths = (char **)list_to_array(slist);

    list *tlist = get_paths(train_images);
    char **tpaths = (char **)list_to_array(tlist);

    load_args targs= get_base_args(gnet);
    targs.paths = tpaths;
    targs.n = imgs;
    targs.m = tlist->size;
    targs.d = &tbuffer;
    targs.type = CLASSIFICATION_DATA;
    targs.classes = 1;
    char *ls[1] = {"zelda"};
    targs.labels = ls;

    load_args sargs = get_base_args(gnet);
    sargs.paths = spaths;
    sargs.n = imgs;
    sargs.m = slist->size;
    sargs.d = &sbuffer;
    sargs.type = CLASSIFICATION_DATA;
    sargs.classes = 1;
    sargs.labels = ls;

    pthread_t tload_thread = load_data_in_thread(targs);
    pthread_t sload_thread = load_data_in_thread(sargs);
    clock_t time;

    float aloss_avg = -1;
    float floss_avg = -1;

    network_state fstate = {};
    fstate.index = 0;
    fstate.net = fnet;
    int x_size = get_network_input_size(fnet)*fnet.batch;
    int y_size = get_network_output_size(fnet)*fnet.batch;
    fstate.input = cuda_make_array(0, x_size);
    fstate.truth = cuda_make_array(0, y_size);
    fstate.delta = cuda_make_array(0, x_size);
    fstate.train = 1;
    float *X = (float*)calloc(x_size, sizeof(float));
    float *y = (float*)calloc(y_size, sizeof(float));

    float *ones = cuda_make_array(0, anet.batch);
    float *zeros = cuda_make_array(0, anet.batch);
    fill_ongpu(anet.batch, .99, ones, 1);
    fill_ongpu(anet.batch, .01, zeros, 1);

    network_state astate = {};
    astate.index = 0;
    astate.net = anet;
    int ax_size = get_network_input_size(anet)*anet.batch;
    int ay_size = get_network_output_size(anet)*anet.batch;
    astate.input = 0;
    astate.truth = ones;
    astate.delta = cuda_make_array(0, ax_size);
    astate.train = 1;

    network_state gstate = {};
    gstate.index = 0;
    gstate.net = gnet;
    int gx_size = get_network_input_size(gnet)*gnet.batch;
    int gy_size = get_network_output_size(gnet)*gnet.batch;
    gstate.input = cuda_make_array(0, gx_size);
    gstate.truth = 0;
    gstate.delta = 0;
    gstate.train = 1;

    while (get_current_batch(gnet) < gnet.max_batches) {
        i += 1;
        time=clock();
        pthread_join(tload_thread, 0);
        pthread_join(sload_thread, 0);
        train = tbuffer;
        style = sbuffer;
        tload_thread = load_data_in_thread(targs);
        sload_thread = load_data_in_thread(sargs);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data generated = copy_data(train);
        time=clock();

        int j, k;
        float floss = 0;
        for(j = 0; j < fnet.subdivisions; ++j){
            layer imlayer = gnet.layers[gnet.n - 1];
            get_next_batch(train, fnet.batch, j*fnet.batch, X, y);

            cuda_push_array(fstate.input, X, x_size);
            cuda_push_array(gstate.input, X, gx_size);
            *gnet.seen += gnet.batch;

            forward_network_gpu(fnet, fstate);
            float *feats = fnet.layers[fnet.n - 2].output_gpu;
            copy_ongpu(y_size, feats, 1, fstate.truth, 1);

            forward_network_gpu(gnet, gstate);
            float *gen = gnet.layers[gnet.n-1].output_gpu;
            copy_ongpu(x_size, gen, 1, fstate.input, 1);

            fill_ongpu(x_size, 0, fstate.delta, 1);
            forward_network_gpu(fnet, fstate);
            backward_network_gpu(fnet, fstate);
            //HERE

            astate.input = gen;
            fill_ongpu(ax_size, 0, astate.delta, 1);
            forward_network_gpu(anet, astate);
            backward_network_gpu(anet, astate);

            float *delta = imlayer.delta_gpu;
            fill_ongpu(x_size, 0, delta, 1);
            scal_ongpu(x_size, 100, astate.delta, 1);
            scal_ongpu(x_size, .00001, fstate.delta, 1);
            axpy_ongpu(x_size, 1, fstate.delta, 1, delta, 1);
            axpy_ongpu(x_size, 1, astate.delta, 1, delta, 1);

            //fill_ongpu(x_size, 0, delta, 1);
            //cuda_push_array(delta, X, x_size);
            //axpy_ongpu(x_size, -1, imlayer.output_gpu, 1, delta, 1);
            //printf("pix error: %f\n", cuda_mag_array(delta, x_size));
            printf("fea error: %f\n", cuda_mag_array(fstate.delta, x_size));
            printf("adv error: %f\n", cuda_mag_array(astate.delta, x_size));
            //axpy_ongpu(x_size, 1, astate.delta, 1, delta, 1);

            backward_network_gpu(gnet, gstate);

            floss += get_network_cost(fnet) /(fnet.subdivisions*fnet.batch);

            cuda_pull_array(imlayer.output_gpu, imlayer.output, x_size);
            for(k = 0; k < gnet.batch; ++k){
                int index = j*gnet.batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, generated.X.vals[index], 1);
                generated.y.vals[index][0] = .01;
            }
        }

/*
        image sim = float_to_image(anet.w, anet.h, anet.c, style.X.vals[j]);
        show_image(sim, "style");
        cvWaitKey(0);
        */

        harmless_update_network_gpu(anet);

        data merge = concat_data(style, generated);
        randomize_data(merge);
        float aloss = train_network(anet, merge);

        update_network_gpu(gnet);

        free_data(merge);
        free_data(train);
        free_data(generated);
        free_data(style);
        if (aloss_avg < 0) aloss_avg = aloss;
        if (floss_avg < 0) floss_avg = floss;
        aloss_avg = aloss_avg*.9 + aloss*.1;
        floss_avg = floss_avg*.9 + floss*.1;

        printf("%d: gen: %f, adv: %f | gen_avg: %f, adv_avg: %f, %f rate, %lf seconds, %d images\n", i, floss, aloss, floss_avg, aloss_avg, get_current_rate(gnet), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, gbase, i);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, gbase);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
#endif
}

void train_pix2pix(char *cfg, char *weight, char *acfg, char *aweight, int clear)
{
#ifdef GPU
    //char *train_images = "/home/pjreddie/data/coco/train1.txt";
    //char *train_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    char *train_images = "/home/pjreddie/data/imagenet/imagenet1k.train.list";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network net = load_network(cfg, weight, clear);
    network anet = load_network(acfg, aweight, clear);

    int i, j, k;
    layer imlayer = {};
    for (i = 0; i < net.n; ++i) {
        if (net.layers[i].out_c == 3) {
            imlayer = net.layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    i = *net.seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;
    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[1] = {"coco"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    network_state gstate = {};
    gstate.index = 0;
    gstate.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = x_size;
    gstate.input = cuda_make_array(0, x_size);
    gstate.truth = cuda_make_array(0, y_size);
    gstate.delta = 0;
    gstate.train = 1;
    float *pixs = (float*)calloc(x_size, sizeof(float));
    float *graypixs = (float*)calloc(x_size, sizeof(float));
    float *y = (float*)calloc(y_size, sizeof(float));

    network_state astate = {};
    astate.index = 0;
    astate.net = anet;
    int ay_size = get_network_output_size(anet)*anet.batch;
    astate.input = 0;
    astate.truth = 0;
    astate.delta = 0;
    astate.train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs);
    float *ones_gpu = cuda_make_array(0, ay_size);
    fill_ongpu(ay_size, .9, ones_gpu, 1);

    float aloss_avg = -1;
    float gloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(net) < net.max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gray = copy_data(train);
        for(j = 0; j < imgs; ++j){
            image gim = float_to_image(net.w, net.h, net.c, gray.X.vals[j]);
            grayscale_image_3c(gim);
            train.y.vals[j][0] = .9;

            image yim = float_to_image(net.w, net.h, net.c, train.X.vals[j]);
            //rgb_to_yuv(yim);
        }
        time=clock();
        float gloss = 0;

        for(j = 0; j < net.subdivisions; ++j){
            get_next_batch(train, net.batch, j*net.batch, pixs, y);
            get_next_batch(gray, net.batch, j*net.batch, graypixs, y);
            cuda_push_array(gstate.input, graypixs, x_size);
            cuda_push_array(gstate.truth, pixs, x_size);
            /*
            image origi = float_to_image(net.w, net.h, 3, pixs);
            image grayi = float_to_image(net.w, net.h, 3, graypixs);
            show_image(grayi, "gray");
            show_image(origi, "orig");
            cvWaitKey(0);
            */
            *net.seen += net.batch;
            forward_network_gpu(net, gstate);

            fill_ongpu(imlayer.outputs, 0, imerror, 1);
            astate.input = imlayer.output_gpu;
            astate.delta = imerror;
            astate.truth = ones_gpu;
            forward_network_gpu(anet, astate);
            backward_network_gpu(anet, astate);

            scal_ongpu(imlayer.outputs, .1, net.layers[net.n-1].delta_gpu, 1);

            backward_network_gpu(net, gstate);

            scal_ongpu(imlayer.outputs, 100, imerror, 1);

            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));
            printf("features %f\n", cuda_mag_array(net.layers[net.n-1].delta_gpu, imlayer.outputs));

            axpy_ongpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);

            gloss += get_network_cost(net) /(net.subdivisions*net.batch);

            cuda_pull_array(imlayer.output_gpu, imlayer.output, x_size);
            for(k = 0; k < net.batch; ++k){
                int index = j*net.batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, gray.X.vals[index], 1);
                gray.y.vals[index][0] = .1;
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gray);
        randomize_data(merge);
        float aloss = train_network(anet, merge);

        update_network_gpu(net);
        update_network_gpu(anet);
        free_data(merge);
        free_data(train);
        free_data(gray);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;
        gloss_avg = gloss_avg*.9 + gloss*.1;

        printf("%d: gen: %f, adv: %f | gen_avg: %f, adv_avg: %f, %f rate, %lf seconds, %d images\n", i, gloss, aloss, gloss_avg, aloss_avg, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
#endif
}

void train_colorizer(char *cfg, char *weight, char *acfg, char *aweight, int clear)
{
#ifdef GPU
    //char *train_images = "/home/pjreddie/data/coco/train1.txt";
    //char *train_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    char *train_images = "/home/pjreddie/data/imagenet/imagenet1k.train.list";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network net = load_network(cfg, weight, clear);
    network anet = load_network(acfg, aweight, clear);

    int i, j, k;
    layer imlayer = {};
    for (i = 0; i < net.n; ++i) {
        if (net.layers[i].out_c == 3) {
            imlayer = net.layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    i = *net.seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;
    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[1] = {"imagenet"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    network_state gstate = {};
    gstate.index = 0;
    gstate.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = x_size;
    gstate.input = cuda_make_array(0, x_size);
    gstate.truth = cuda_make_array(0, y_size);
    gstate.delta = 0;
    gstate.train = 1;
    float *pixs = (float*)calloc(x_size, sizeof(float));
    float *graypixs = (float*)calloc(x_size, sizeof(float));
    float *y = (float*)calloc(y_size, sizeof(float));

    network_state astate = {};
    astate.index = 0;
    astate.net = anet;
    int ay_size = get_network_output_size(anet)*anet.batch;
    astate.input = 0;
    astate.truth = 0;
    astate.delta = 0;
    astate.train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs);
    float *ones_gpu = cuda_make_array(0, ay_size);
    fill_ongpu(ay_size, .99, ones_gpu, 1);

    float aloss_avg = -1;
    float gloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(net) < net.max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gray = copy_data(train);
        for(j = 0; j < imgs; ++j){
            image gim = float_to_image(net.w, net.h, net.c, gray.X.vals[j]);
            grayscale_image_3c(gim);
            train.y.vals[j][0] = .99;

            image yim = float_to_image(net.w, net.h, net.c, train.X.vals[j]);
            //rgb_to_yuv(yim);
        }
        time=clock();
        float gloss = 0;

        for(j = 0; j < net.subdivisions; ++j){
            get_next_batch(train, net.batch, j*net.batch, pixs, y);
            get_next_batch(gray, net.batch, j*net.batch, graypixs, y);
            cuda_push_array(gstate.input, graypixs, x_size);
            cuda_push_array(gstate.truth, pixs, x_size);
            /*
            image origi = float_to_image(net.w, net.h, 3, pixs);
            image grayi = float_to_image(net.w, net.h, 3, graypixs);
            show_image(grayi, "gray");
            show_image(origi, "orig");
            cvWaitKey(0);
            */
            *net.seen += net.batch;
            forward_network_gpu(net, gstate);

            fill_ongpu(imlayer.outputs, 0, imerror, 1);
            astate.input = imlayer.output_gpu;
            astate.delta = imerror;
            astate.truth = ones_gpu;
            forward_network_gpu(anet, astate);
            backward_network_gpu(anet, astate);

            scal_ongpu(imlayer.outputs, .1, net.layers[net.n-1].delta_gpu, 1);

            backward_network_gpu(net, gstate);

            scal_ongpu(imlayer.outputs, 100, imerror, 1);

            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));
            printf("features %f\n", cuda_mag_array(net.layers[net.n-1].delta_gpu, imlayer.outputs));

            axpy_ongpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);

            gloss += get_network_cost(net) /(net.subdivisions*net.batch);

            cuda_pull_array(imlayer.output_gpu, imlayer.output, x_size);
            for(k = 0; k < net.batch; ++k){
                int index = j*net.batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, gray.X.vals[index], 1);
                gray.y.vals[index][0] = .01;
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gray);
        randomize_data(merge);
        float aloss = train_network(anet, merge);

        update_network_gpu(net);
        update_network_gpu(anet);
        free_data(merge);
        free_data(train);
        free_data(gray);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;
        gloss_avg = gloss_avg*.9 + gloss*.1;

        printf("%d: gen: %f, adv: %f | gen_avg: %f, adv_avg: %f, %f rate, %lf seconds, %d images\n", i, gloss, aloss, gloss_avg, aloss_avg, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
#endif
}

void train_lsd2(char *cfgfile, char *weightfile, char *acfgfile, char *aweightfile, int clear)
{
#ifdef GPU
    char *train_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    if(clear) *net.seen = 0;

    char *abase = basecfg(acfgfile);
    network anet = parse_network_cfg(acfgfile);
    if(aweightfile){
        load_weights(&anet, aweightfile);
    }
    if(clear) *anet.seen = 0;

    int i, j, k;
    layer imlayer = {};
    for (i = 0; i < net.n; ++i) {
        if (net.layers[i].out_c == 3) {
            imlayer = net.layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    i = *net.seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;
    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[1] = {"coco"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    network_state gstate = {};
    gstate.index = 0;
    gstate.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = 1*net.batch;
    gstate.input = cuda_make_array(0, x_size);
    gstate.truth = 0;
    gstate.delta = 0;
    gstate.train = 1;
    float *X = (float*)calloc(x_size, sizeof(float));
    float *y = (float*)calloc(y_size, sizeof(float));

    network_state astate = {};
    astate.index = 0;
    astate.net = anet;
    int ay_size = get_network_output_size(anet)*anet.batch;
    astate.input = 0;
    astate.truth = 0;
    astate.delta = 0;
    astate.train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs);
    float *ones_gpu = cuda_make_array(0, ay_size);
    fill_ongpu(ay_size, 1, ones_gpu, 1);

    float aloss_avg = -1;
    float gloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(net) < net.max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data generated = copy_data(train);
        time=clock();
        float gloss = 0;

        for(j = 0; j < net.subdivisions; ++j){
            get_next_batch(train, net.batch, j*net.batch, X, y);
            cuda_push_array(gstate.input, X, x_size);
            *net.seen += net.batch;
            forward_network_gpu(net, gstate);

            fill_ongpu(imlayer.outputs, 0, imerror, 1);
            astate.input = imlayer.output_gpu;
            astate.delta = imerror;
            astate.truth = ones_gpu;
            forward_network_gpu(anet, astate);
            backward_network_gpu(anet, astate);

            scal_ongpu(imlayer.outputs, 1, imerror, 1);
            axpy_ongpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);

            backward_network_gpu(net, gstate);

            printf("features %f\n", cuda_mag_array(imlayer.delta_gpu, imlayer.outputs));
            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));

            gloss += get_network_cost(net) /(net.subdivisions*net.batch);

            cuda_pull_array(imlayer.output_gpu, imlayer.output, x_size);
            for(k = 0; k < net.batch; ++k){
                int index = j*net.batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, generated.X.vals[index], 1);
                generated.y.vals[index][0] = 0;
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, generated);
        randomize_data(merge);
        float aloss = train_network(anet, merge);

        update_network_gpu(net);
        update_network_gpu(anet);
        free_data(merge);
        free_data(train);
        free_data(generated);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;
        gloss_avg = gloss_avg*.9 + gloss*.1;

        printf("%d: gen: %f, adv: %f | gen_avg: %f, adv_avg: %f, %f rate, %lf seconds, %d images\n", i, gloss, aloss, gloss_avg, aloss_avg, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
#endif
}

void train_lsd(char *cfgfile, char *weightfile, int clear)
{
    char *train_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    if(clear) *net.seen = 0;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;
    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[1] = {"coco"};
    args.labels = ls;

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

void test_lsd(char *cfgfile, char *weightfile, char *filename)
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
    int i, imlayer = 0;

    for (i = 0; i < net.n; ++i) {
        if (net.layers[i].out_c == 3) {
            imlayer = i;
            printf("%d\n", i);
            break;
        }
    }

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
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        //grayscale_image_3c(crop);

        float *X = crop.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);
        //yuv_to_rgb(out);
        constrain_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        show_image(out, "out");
        show_image(crop, "crop");
        save_image(out, "out");
#ifdef OPENCV
        cvWaitKey(0);
#endif

        free_image(im);
        free_image(resized);
        free_image(crop);
        if (filename) break;
    }
}


void run_lsd(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    char *acfg = argv[5];
    char *aweights = (argc > 6) ? argv[6] : 0;
    if(0==strcmp(argv[2], "train")) train_lsd(cfg, weights, clear);
    else if(0==strcmp(argv[2], "train2")) train_lsd2(cfg, weights, acfg, aweights, clear);
    else if(0==strcmp(argv[2], "traincolor")) train_colorizer(cfg, weights, acfg, aweights, clear);
    else if(0==strcmp(argv[2], "train3")) train_lsd3(argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], clear);
    else if(0==strcmp(argv[2], "test")) test_lsd(cfg, weights, filename);
    /*
       else if(0==strcmp(argv[2], "valid")) validate_lsd(cfg, weights);
     */
}
