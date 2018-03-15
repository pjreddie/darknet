#include "darknet.h"

/*
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

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", gnet->learning_rate, gnet->momentum, gnet->decay);
    int imgs = gnet->batch*gnet->subdivisions;
    int i = *gnet->seen/imgs;
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

    fnet->train=1;
    int x_size = fnet->inputs*fnet->batch;
    int y_size = fnet->truths*fnet->batch;
    float *X = calloc(x_size, sizeof(float));
    float *y = calloc(y_size, sizeof(float));


    int ax_size = anet->inputs*anet->batch;
    int ay_size = anet->truths*anet->batch;
    fill_gpu(ay_size, .9, anet->truth_gpu, 1);
    anet->delta_gpu = cuda_make_array(0, ax_size);
    anet->train = 1;

    int gx_size = gnet->inputs*gnet->batch;
    int gy_size = gnet->truths*gnet->batch;
    gstate.input = cuda_make_array(0, gx_size);
    gstate.truth = 0;
    gstate.delta = 0;
    gstate.train = 1;

    while (get_current_batch(gnet) < gnet->max_batches) {
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
        for(j = 0; j < fnet->subdivisions; ++j){
            layer imlayer = gnet->layers[gnet->n - 1];
            get_next_batch(train, fnet->batch, j*fnet->batch, X, y);

            cuda_push_array(fstate.input, X, x_size);
            cuda_push_array(gstate.input, X, gx_size);
            *gnet->seen += gnet->batch;

            forward_network_gpu(fnet, fstate);
            float *feats = fnet->layers[fnet->n - 2].output_gpu;
            copy_gpu(y_size, feats, 1, fstate.truth, 1);

            forward_network_gpu(gnet, gstate);
            float *gen = gnet->layers[gnet->n-1].output_gpu;
            copy_gpu(x_size, gen, 1, fstate.input, 1);

            fill_gpu(x_size, 0, fstate.delta, 1);
            forward_network_gpu(fnet, fstate);
            backward_network_gpu(fnet, fstate);
            //HERE

            astate.input = gen;
            fill_gpu(ax_size, 0, astate.delta, 1);
            forward_network_gpu(anet, astate);
            backward_network_gpu(anet, astate);

            float *delta = imlayer.delta_gpu;
            fill_gpu(x_size, 0, delta, 1);
            scal_gpu(x_size, 100, astate.delta, 1);
            scal_gpu(x_size, .001, fstate.delta, 1);
            axpy_gpu(x_size, 1, fstate.delta, 1, delta, 1);
            axpy_gpu(x_size, 1, astate.delta, 1, delta, 1);

            //fill_gpu(x_size, 0, delta, 1);
            //cuda_push_array(delta, X, x_size);
            //axpy_gpu(x_size, -1, imlayer.output_gpu, 1, delta, 1);
            //printf("pix error: %f\n", cuda_mag_array(delta, x_size));
            printf("fea error: %f\n", cuda_mag_array(fstate.delta, x_size));
            printf("adv error: %f\n", cuda_mag_array(astate.delta, x_size));
            //axpy_gpu(x_size, 1, astate.delta, 1, delta, 1);

            backward_network_gpu(gnet, gstate);

            floss += get_network_cost(fnet) /(fnet->subdivisions*fnet->batch);

            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
            for(k = 0; k < gnet->batch; ++k){
                int index = j*gnet->batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, generated.X.vals[index], 1);
                generated.y.vals[index][0] = .1;
                style.y.vals[index][0] = .9;
            }
        }

*/
/*
        image sim = float_to_image(anet->w, anet->h, anet->c, style.X.vals[j]);
        show_image(sim, "style");
        cvWaitKey(0);
        */
        /*

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
*/

/*
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
    layer imlayer = {0};
    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = net->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[1] = {"coco"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    network_state gstate = {0};
    gstate.index = 0;
    gstate.net = net;
    int x_size = get_network_input_size(net)*net->batch;
    int y_size = x_size;
    gstate.input = cuda_make_array(0, x_size);
    gstate.truth = cuda_make_array(0, y_size);
    gstate.delta = 0;
    gstate.train = 1;
    float *pixs = calloc(x_size, sizeof(float));
    float *graypixs = calloc(x_size, sizeof(float));
    float *y = calloc(y_size, sizeof(float));

    network_state astate = {0};
    astate.index = 0;
    astate.net = anet;
    int ay_size = get_network_output_size(anet)*anet->batch;
    astate.input = 0;
    astate.truth = 0;
    astate.delta = 0;
    astate.train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs);
    float *ones_gpu = cuda_make_array(0, ay_size);
    fill_gpu(ay_size, .9, ones_gpu, 1);

    float aloss_avg = -1;
    float gloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(net) < net->max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gray = copy_data(train);
        for(j = 0; j < imgs; ++j){
            image gim = float_to_image(net->w, net->h, net->c, gray.X.vals[j]);
            grayscale_image_3c(gim);
            train.y.vals[j][0] = .9;

            image yim = float_to_image(net->w, net->h, net->c, train.X.vals[j]);
            //rgb_to_yuv(yim);
        }
        time=clock();
        float gloss = 0;

        for(j = 0; j < net->subdivisions; ++j){
            get_next_batch(train, net->batch, j*net->batch, pixs, y);
            get_next_batch(gray, net->batch, j*net->batch, graypixs, y);
            cuda_push_array(gstate.input, graypixs, x_size);
            cuda_push_array(gstate.truth, pixs, y_size);
            */
            /*
            image origi = float_to_image(net->w, net->h, 3, pixs);
            image grayi = float_to_image(net->w, net->h, 3, graypixs);
            show_image(grayi, "gray");
            show_image(origi, "orig");
            cvWaitKey(0);
            */
            /*
            *net->seen += net->batch;
            forward_network_gpu(net, gstate);

            fill_gpu(imlayer.outputs, 0, imerror, 1);
            astate.input = imlayer.output_gpu;
            astate.delta = imerror;
            astate.truth = ones_gpu;
            forward_network_gpu(anet, astate);
            backward_network_gpu(anet, astate);

            scal_gpu(imlayer.outputs, .1, net->layers[net->n-1].delta_gpu, 1);

            backward_network_gpu(net, gstate);

            scal_gpu(imlayer.outputs, 1000, imerror, 1);

            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));
            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs));

            axpy_gpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);

            gloss += get_network_cost(net) /(net->subdivisions*net->batch);

            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
            for(k = 0; k < net->batch; ++k){
                int index = j*net->batch + k;
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
*/

void test_dcgan(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    int i, imlayer = 0;

    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = i;
            printf("%d\n", i);
            break;
        }
    }

    while(1){
        image im = make_image(net->w, net->h, net->c);
        int i;
        for(i = 0; i < im.w*im.h*im.c; ++i){
            im.data[i] = rand_normal();
        }

        float *X = im.data;
        time=clock();
        network_predict(net, X);
        image out = get_network_image_layer(net, imlayer);
        //yuv_to_rgb(out);
        normalize_image(out);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        show_image(out, "out");
        save_image(out, "out");
#ifdef OPENCV
        cvWaitKey(0);
#endif

        free_image(im);
    }
}

void dcgan_batch(network gnet, network anet)
{
    //float *input = calloc(x_size, sizeof(float));
}


void train_dcgan(char *cfg, char *weight, char *acfg, char *aweight, int clear, int display, char *train_images)
{
#ifdef GPU
    //char *train_images = "/home/pjreddie/data/coco/train1.txt";
    //char *train_images = "/home/pjreddie/data/coco/trainvalno5k.txt";
    //char *train_images = "/home/pjreddie/data/imagenet/imagenet1k.train.list";
    //char *train_images = "data/64.txt";
    //char *train_images = "data/alp.txt";
    //char *train_images = "data/cifar.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    char *base = basecfg(cfg);
    char *abase = basecfg(acfg);
    printf("%s\n", base);
    network *gnet = load_network(cfg, weight, clear);
    network *anet = load_network(acfg, aweight, clear);
    //float orig_rate = anet->learning_rate;

    int start = 0;
    int i, j, k;
    layer imlayer = {0};
    for (i = 0; i < gnet->n; ++i) {
        if (gnet->layers[i].out_c == 3) {
            imlayer = gnet->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", gnet->learning_rate, gnet->momentum, gnet->decay);
    int imgs = gnet->batch*gnet->subdivisions;
    i = *gnet->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(anet);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;
    args.threads=16;
    args.classes = 1;
    char *ls[2] = {"imagenet", "zzzzzzzz"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    gnet->train = 1;
    anet->train = 1;

    int x_size = gnet->inputs*gnet->batch;
    int y_size = gnet->truths*gnet->batch;
    float *imerror = cuda_make_array(0, y_size);

    //int ay_size = anet->truths*anet->batch;

    float aloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(gnet) < gnet->max_batches) {
    start += 1;
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        //translate_data_rows(train, -.5);
        //scale_data_rows(train, 2);

        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gen = copy_data(train);
        for (j = 0; j < imgs; ++j) {
            train.y.vals[j][0] = .95;
            gen.y.vals[j][0] = .05;
        }
        time=clock();

        for(j = 0; j < gnet->subdivisions; ++j){
            get_next_batch(train, gnet->batch, j*gnet->batch, gnet->truth, 0);
            int z;
            for(z = 0; z < x_size; ++z){
                gnet->input[z] = rand_normal();
            }

            cuda_push_array(gnet->input_gpu, gnet->input, x_size);
            cuda_push_array(gnet->truth_gpu, gnet->truth, y_size);
            *gnet->seen += gnet->batch;
            forward_network_gpu(gnet);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            fill_gpu(anet->truths*anet->batch, .95, anet->truth_gpu, 1);
            copy_gpu(anet->inputs*anet->batch, imlayer.output_gpu, 1, anet->input_gpu, 1);
            anet->delta_gpu = imerror;
            forward_network_gpu(anet);
            backward_network_gpu(anet);

            float genaloss = *anet->cost / anet->batch;
            printf("%f\n", genaloss);

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);
            scal_gpu(imlayer.outputs*imlayer.batch, .00, gnet->layers[gnet->n-1].delta_gpu, 1);

            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
            printf("features %f\n", cuda_mag_array(gnet->layers[gnet->n-1].delta_gpu, imlayer.outputs*imlayer.batch));

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, gnet->layers[gnet->n-1].delta_gpu, 1);

            backward_network_gpu(gnet);

            for(k = 0; k < gnet->batch; ++k){
                int index = j*gnet->batch + k;
                copy_cpu(gnet->outputs, gnet->output + k*gnet->outputs, 1, gen.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gen);
        //randomize_data(merge);
        float aloss = train_network(anet, merge);

        //translate_image(im, 1);
        //scale_image(im, .5);
        //translate_image(im2, 1);
        //scale_image(im2, .5);
            #ifdef OPENCV
        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gen.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen");
            show_image(im2, "train");
            cvWaitKey(50);
        }
        #endif

/*
        if(aloss < .1){
            anet->learning_rate = 0;
        } else if (aloss > .3){
            anet->learning_rate = orig_rate;
        }
        */

        update_network_gpu(gnet);

        free_data(merge);
        free_data(train);
        free_data(gen);
        if (aloss_avg < 0) aloss_avg = aloss;
        aloss_avg = aloss_avg*.9 + aloss*.1;

        printf("%d: adv: %f | adv_avg: %f, %f rate, %lf seconds, %d images\n", i, aloss, aloss_avg, get_current_rate(gnet), sec(clock()-time), i*imgs);
        if(i%10000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, abase, i);
            save_weights(anet, buff);
        }
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(gnet, buff);
            sprintf(buff, "%s/%s.backup", backup_directory, abase);
            save_weights(anet, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(gnet, buff);
#endif
}

void train_colorizer(char *cfg, char *weight, char *acfg, char *aweight, int clear, int display)
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
    network *net = load_network(cfg, weight, clear);
    network *anet = load_network(acfg, aweight, clear);

    int i, j, k;
    layer imlayer = {0};
    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = net->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args= get_base_args(net);
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[2] = {"imagenet"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    int x_size = net->inputs*net->batch;
    //int y_size = x_size;
    net->delta = 0;
    net->train = 1;
    float *pixs = calloc(x_size, sizeof(float));
    float *graypixs = calloc(x_size, sizeof(float));
    //float *y = calloc(y_size, sizeof(float));

    //int ay_size = anet->outputs*anet->batch;
    anet->delta = 0;
    anet->train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs*imlayer.batch);

    float aloss_avg = -1;
    float gloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(net) < net->max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data gray = copy_data(train);
        for(j = 0; j < imgs; ++j){
            image gim = float_to_image(net->w, net->h, net->c, gray.X.vals[j]);
            grayscale_image_3c(gim);
            train.y.vals[j][0] = .95;
            gray.y.vals[j][0] = .05;
        }
        time=clock();
        float gloss = 0;

        for(j = 0; j < net->subdivisions; ++j){
            get_next_batch(train, net->batch, j*net->batch, pixs, 0);
            get_next_batch(gray, net->batch, j*net->batch, graypixs, 0);
            cuda_push_array(net->input_gpu, graypixs, net->inputs*net->batch);
            cuda_push_array(net->truth_gpu, pixs, net->truths*net->batch);
            /*
               image origi = float_to_image(net->w, net->h, 3, pixs);
               image grayi = float_to_image(net->w, net->h, 3, graypixs);
               show_image(grayi, "gray");
               show_image(origi, "orig");
               cvWaitKey(0);
             */
            *net->seen += net->batch;
            forward_network_gpu(net);

            fill_gpu(imlayer.outputs*imlayer.batch, 0, imerror, 1);
            copy_gpu(anet->inputs*anet->batch, imlayer.output_gpu, 1, anet->input_gpu, 1);
            fill_gpu(anet->inputs*anet->batch, .95, anet->truth_gpu, 1);
            anet->delta_gpu = imerror;
            forward_network_gpu(anet);
            backward_network_gpu(anet);

            scal_gpu(imlayer.outputs*imlayer.batch, 1./100., net->layers[net->n-1].delta_gpu, 1);

            scal_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1);

            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs*imlayer.batch));
            printf("features %f\n", cuda_mag_array(net->layers[net->n-1].delta_gpu, imlayer.outputs*imlayer.batch));

            axpy_gpu(imlayer.outputs*imlayer.batch, 1, imerror, 1, net->layers[net->n-1].delta_gpu, 1);

            backward_network_gpu(net);


            gloss += *net->cost /(net->subdivisions*net->batch);

            for(k = 0; k < net->batch; ++k){
                int index = j*net->batch + k;
                copy_cpu(imlayer.outputs, imlayer.output + k*imlayer.outputs, 1, gray.X.vals[index], 1);
            }
        }
        harmless_update_network_gpu(anet);

        data merge = concat_data(train, gray);
        //randomize_data(merge);
        float aloss = train_network(anet, merge);

        update_network_gpu(net);

            #ifdef OPENCV
        if(display){
            image im = float_to_image(anet->w, anet->h, anet->c, gray.X.vals[0]);
            image im2 = float_to_image(anet->w, anet->h, anet->c, train.X.vals[0]);
            show_image(im, "gen");
            show_image(im2, "train");
            cvWaitKey(50);
        }
        #endif
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

/*
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
    if(clear) *net->seen = 0;

    char *abase = basecfg(acfgfile);
    network anet = parse_network_cfg(acfgfile);
    if(aweightfile){
        load_weights(&anet, aweightfile);
    }
    if(clear) *anet->seen = 0;

    int i, j, k;
    layer imlayer = {0};
    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
            imlayer = net->layers[i];
            break;
        }
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[1] = {"coco"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

    network_state gstate = {0};
    gstate.index = 0;
    gstate.net = net;
    int x_size = get_network_input_size(net)*net->batch;
    int y_size = 1*net->batch;
    gstate.input = cuda_make_array(0, x_size);
    gstate.truth = 0;
    gstate.delta = 0;
    gstate.train = 1;
    float *X = calloc(x_size, sizeof(float));
    float *y = calloc(y_size, sizeof(float));

    network_state astate = {0};
    astate.index = 0;
    astate.net = anet;
    int ay_size = get_network_output_size(anet)*anet->batch;
    astate.input = 0;
    astate.truth = 0;
    astate.delta = 0;
    astate.train = 1;

    float *imerror = cuda_make_array(0, imlayer.outputs);
    float *ones_gpu = cuda_make_array(0, ay_size);
    fill_gpu(ay_size, 1, ones_gpu, 1);

    float aloss_avg = -1;
    float gloss_avg = -1;

    //data generated = copy_data(train);

    while (get_current_batch(net) < net->max_batches) {
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        data generated = copy_data(train);
        time=clock();
        float gloss = 0;

        for(j = 0; j < net->subdivisions; ++j){
            get_next_batch(train, net->batch, j*net->batch, X, y);
            cuda_push_array(gstate.input, X, x_size);
            *net->seen += net->batch;
            forward_network_gpu(net, gstate);

            fill_gpu(imlayer.outputs, 0, imerror, 1);
            astate.input = imlayer.output_gpu;
            astate.delta = imerror;
            astate.truth = ones_gpu;
            forward_network_gpu(anet, astate);
            backward_network_gpu(anet, astate);

            scal_gpu(imlayer.outputs, 1, imerror, 1);
            axpy_gpu(imlayer.outputs, 1, imerror, 1, imlayer.delta_gpu, 1);

            backward_network_gpu(net, gstate);

            printf("features %f\n", cuda_mag_array(imlayer.delta_gpu, imlayer.outputs));
            printf("realness %f\n", cuda_mag_array(imerror, imlayer.outputs));

            gloss += get_network_cost(net) /(net->subdivisions*net->batch);

            cuda_pull_array(imlayer.output_gpu, imlayer.output, imlayer.outputs*imlayer.batch);
            for(k = 0; k < net->batch; ++k){
                int index = j*net->batch + k;
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
*/

/*
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
    if(clear) *net->seen = 0;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int imgs = net->batch*net->subdivisions;
    int i = *net->seen/imgs;
    data train, buffer;


    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    args.type = CLASSIFICATION_DATA;
    args.classes = 1;
    char *ls[1] = {"coco"};
    args.labels = ls;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
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
*/

void test_lsd(char *cfg, char *weights, char *filename, int gray)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    int i, imlayer = 0;

    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].out_c == 3) {
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
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        if(gray) grayscale_image_3c(crop);

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
    int display = find_arg(argc, argv, "-display");
    char *file = find_char_arg(argc, argv, "-file", "/home/pjreddie/data/imagenet/imagenet1k.train.list");

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    char *acfg = argv[5];
    char *aweights = (argc > 6) ? argv[6] : 0;
    //if(0==strcmp(argv[2], "train")) train_lsd(cfg, weights, clear);
    //else if(0==strcmp(argv[2], "train2")) train_lsd2(cfg, weights, acfg, aweights, clear);
    //else if(0==strcmp(argv[2], "traincolor")) train_colorizer(cfg, weights, acfg, aweights, clear);
    //else if(0==strcmp(argv[2], "train3")) train_lsd3(argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], clear);
    if(0==strcmp(argv[2], "traingan")) train_dcgan(cfg, weights, acfg, aweights, clear, display, file);
    else if(0==strcmp(argv[2], "traincolor")) train_colorizer(cfg, weights, acfg, aweights, clear, display);
    else if(0==strcmp(argv[2], "gan")) test_dcgan(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_lsd(cfg, weights, filename, 0);
    else if(0==strcmp(argv[2], "color")) test_lsd(cfg, weights, filename, 1);
    /*
       else if(0==strcmp(argv[2], "valid")) validate_lsd(cfg, weights);
     */
}
