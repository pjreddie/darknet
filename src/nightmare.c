
#include "network.h"
#include "parser.h"
#include "blas.h"
#include "utils.h"

float abs_mean(float *x, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i){
        sum += abs(x[i]);
    }
    return sum/n;
}

void calculate_loss(float *output, float *delta, int n, float thresh)
{
    int i;
    float mean = mean_array(output, n); 
    float var = variance_array(output, n);
    for(i = 0; i < n; ++i){
        if(delta[i] > mean + thresh*sqrt(var)) delta[i] = output[i];
        else delta[i] = 0;
    }
}

void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm)
{
    scale_image(orig, 2);
    translate_image(orig, -1);
    net->n = max_layer + 1;

    int dx = rand()%16 - 8;
    int dy = rand()%16 - 8;
    int flip = rand()%2;

    image crop = crop_image(orig, dx, dy, orig.w, orig.h);
    image im = resize_image(crop, (int)(orig.w * scale), (int)(orig.h * scale));
    if(flip) flip_image(im);

    resize_network(net, im.w, im.h);
    layer last = net->layers[net->n-1];
    //net->layers[net->n - 1].activation = LINEAR;

    image delta = make_image(im.w, im.h, im.c);

    network_state state = {0};

#ifdef GPU
    state.input = cuda_make_array(im.data, im.w*im.h*im.c);
    state.delta = cuda_make_array(im.data, im.w*im.h*im.c);

    forward_network_gpu(*net, state);
    copy_ongpu(last.outputs, last.output_gpu, 1, last.delta_gpu, 1);

    cuda_pull_array(last.delta_gpu, last.delta, last.outputs);
    calculate_loss(last.delta, last.delta, last.outputs, thresh);
    cuda_push_array(last.delta_gpu, last.delta, last.outputs);

    backward_network_gpu(*net, state);

    cuda_pull_array(state.delta, delta.data, im.w*im.h*im.c);
    cuda_free(state.input);
    cuda_free(state.delta);
#else
    state.input = im.data;
    state.delta = delta.data;
    forward_network(*net, state);
    copy_cpu(last.outputs, last.output, 1, last.delta, 1);
    calculate_loss(last.output, last.delta, last.outputs, thresh);
    backward_network(*net, state);
#endif

    if(flip) flip_image(delta);
    //normalize_array(delta.data, delta.w*delta.h*delta.c);
    image resized = resize_image(delta, orig.w, orig.h);
    image out = crop_image(resized, -dx, -dy, orig.w, orig.h);

    /*
       image g = grayscale_image(out);
       free_image(out);
       out = g;
     */

    //rate = rate / abs_mean(out.data, out.w*out.h*out.c);

    if(norm) normalize_array(out.data, out.w*out.h*out.c);
    axpy_cpu(orig.w*orig.h*orig.c, rate, out.data, 1, orig.data, 1);

    /*
       normalize_array(orig.data, orig.w*orig.h*orig.c);
       scale_image(orig, sqrt(var));
       translate_image(orig, mean);
     */

    translate_image(orig, 1);
    scale_image(orig, .5);
    //normalize_image(orig);

    constrain_image(orig);

    free_image(crop);
    free_image(im);
    free_image(delta);
    free_image(resized);
    free_image(out);

}


void run_nightmare(int argc, char **argv)
{
    srand(0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [cfg] [weights] [image] [layer] [options! (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[2];
    char *weights = argv[3];
    char *input = argv[4];
    int max_layer = atoi(argv[5]);

    int range = find_int_arg(argc, argv, "-range", 1);
    int norm = find_int_arg(argc, argv, "-norm", 1);
    int rounds = find_int_arg(argc, argv, "-rounds", 1);
    int iters = find_int_arg(argc, argv, "-iters", 10);
    int octaves = find_int_arg(argc, argv, "-octaves", 4);
    float zoom = find_float_arg(argc, argv, "-zoom", 1.);
    float rate = find_float_arg(argc, argv, "-rate", .04);
    float thresh = find_float_arg(argc, argv, "-thresh", 1.);
    float rotate = find_float_arg(argc, argv, "-rotate", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);

    network net = parse_network_cfg(cfg);
    load_weights(&net, weights);
    char *cfgbase = basecfg(cfg);
    char *imbase = basecfg(input);

    set_batch_network(&net, 1);
    image im = load_image_color(input, 0, 0);
    if(0){
        float scale = 1;
        if(im.w > 512 || im.h > 512){
            if(im.w > im.h) scale = 512.0/im.w;
            else scale = 512.0/im.h;
        }
        image resized = resize_image(im, scale*im.w, scale*im.h);
        free_image(im);
        im = resized;
    }

    int e;
    int n;
    for(e = 0; e < rounds; ++e){
            fprintf(stderr, "Iteration: ");
            fflush(stderr);
        for(n = 0; n < iters; ++n){  
            fprintf(stderr, "%d, ", n);
            fflush(stderr);
            int layer = max_layer + rand()%range - range/2;
            int octave = rand()%octaves;
            optimize_picture(&net, im, layer, 1/pow(1.33333333, octave), rate, thresh, norm);
        }
        fprintf(stderr, "done\n");
        if(0){
            image g = grayscale_image(im);
            free_image(im);
            im = g;
        }
        char buff[256];
        if (prefix){
            sprintf(buff, "%s/%s_%s_%d_%06d",prefix, imbase, cfgbase, max_layer, e);
        }else{
            sprintf(buff, "%s_%s_%d_%06d",imbase, cfgbase, max_layer, e);
        }
        printf("%d %s\n", e, buff);
        save_image(im, buff);
        //show_image(im, buff);
        //cvWaitKey(0);

        if(rotate){
            image rot = rotate_image(im, rotate);
            free_image(im);
            im = rot;
        }
        image crop = crop_image(im, im.w * (1. - zoom)/2., im.h * (1.-zoom)/2., im.w*zoom, im.h*zoom);
        image resized = resize_image(crop, im.w, im.h);
        free_image(im);
        free_image(crop);
        im = resized;
    }
}

