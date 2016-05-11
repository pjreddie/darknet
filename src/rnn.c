#include "network.h"
#include "cost_layer.h"
#include "utils.h"
#include "blas.h"
#include "parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

typedef struct {
    float *x;
    float *y;
} float_pair;

float_pair get_rnn_data(unsigned char *text, size_t *offsets, int characters, size_t len, int batch, int steps)
{
    float *x = calloc(batch * steps * characters, sizeof(float));
    float *y = calloc(batch * steps * characters, sizeof(float));
    int i,j;
    for(i = 0; i < batch; ++i){
        for(j = 0; j < steps; ++j){
            unsigned char curr = text[(offsets[i])%len];
            unsigned char next = text[(offsets[i] + 1)%len];

            x[(j*batch + i)*characters + curr] = 1;
            y[(j*batch + i)*characters + next] = 1;

            offsets[i] = (offsets[i] + 1) % len;

            if(curr > 255 || curr <= 0 || next > 255 || next <= 0){
                /*text[(index+j+2)%len] = 0;
                printf("%ld %d %d %d %d\n", index, j, len, (int)text[index+j], (int)text[index+j+1]);
                printf("%s", text+index);
                */
                error("Bad char");
            }
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}

void reset_rnn_state(network net, int b)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        #ifdef GPU
        if(l.state_gpu){
            fill_ongpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        #endif
    }
}

void train_char_rnn(char *cfgfile, char *weightfile, char *filename, int clear)
{
    srand(time(0));
    data_seed = time(0);
    FILE *fp = fopen(filename, "rb");

    fseek(fp, 0, SEEK_END); 
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET); 

    unsigned char *text = calloc(size+1, sizeof(char));
    fread(text, 1, size, fp);
    fclose(fp);

    char *backup_directory = "/home/pjreddie/backup/";
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }

    int inputs = get_network_input_size(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int batch = net.batch;
    int steps = net.time_steps;
    if(clear) *net.seen = 0;
    int i = (*net.seen)/net.batch;

    int streams = batch/steps;
    size_t *offsets = calloc(streams, sizeof(size_t));
    int j;
    for(j = 0; j < streams; ++j){
        offsets[j] = rand_size_t()%size;
    }

    clock_t time;
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        float_pair p = get_rnn_data(text, offsets, inputs, size, streams, steps);

        float loss = train_network_datum(net, p.x, p.y) / (batch);
        free(p.x);
        free(p.y);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        int chars = get_current_batch(net)*batch;
        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds, %f epochs\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), (float) chars/size);

        for(j = 0; j < streams; ++j){
            //printf("%d\n", j);
            if(rand()%10 == 0){
                //fprintf(stderr, "Reset\n");
                offsets[j] = rand_size_t()%size;
                reset_rnn_state(net, j);
            }
        }

        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%10==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void test_char_rnn(char *cfgfile, char *weightfile, int num, char *seed, float temp, int rseed)
{
    srand(rseed);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int inputs = get_network_input_size(net);

    int i, j;
    for(i = 0; i < net.n; ++i) net.layers[i].temperature = temp;
    unsigned char c;
    int len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));

/*
    fill_cpu(inputs, 0, input, 1);
    for(i = 0; i < 10; ++i){
        network_predict(net, input);
    }
    fill_cpu(inputs, 0, input, 1);
    */

    for(i = 0; i < len-1; ++i){
        c = seed[i];
        input[(int)c] = 1;
        network_predict(net, input);
        input[(int)c] = 0;
        printf("%c", c);
    }
    c = seed[len-1];
    for(i = 0; i < num; ++i){
        printf("%c", c);
        input[(int)c] = 1;
        float *out = network_predict(net, input);
        input[(int)c] = 0;
        for(j = 32; j < 127; ++j){
            //printf("%d %c %f\n",j, j, out[j]);
        }
        for(j = 0; j < inputs; ++j){
            //if (out[j] < .0001) out[j] = 0;
        }
        c = sample_array(out, inputs);
    }
    printf("\n");
}

void valid_char_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int inputs = get_network_input_size(net);

    int count = 0;
    int c;
    int len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));
    int i;
    for(i = 0; i < len; ++i){
        c = seed[i];
        input[(int)c] = 1;
        network_predict(net, input);
        input[(int)c] = 0;
    }
    float sum = 0;
    c = getc(stdin);
    float log2 = log(2);
    while(c != EOF){
        int next = getc(stdin);
        if(next == EOF) break;
        if(next < 0 || next >= 255) error("Out of range character");
        ++count;
        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;
        sum += log(out[next])/log2;
        c = next;
        printf("%d Perplexity: %f\n", count, pow(2, -sum/count));
    }
}

void vec_char_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int inputs = get_network_input_size(net);

    int c;
    int seed_len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));
    int i;
    char *line;
    while((line=fgetl(stdin)) != 0){
        reset_rnn_state(net, 0);
        for(i = 0; i < seed_len; ++i){
            c = seed[i];
            input[(int)c] = 1;
            network_predict(net, input);
            input[(int)c] = 0;
        }
        strip(line);
        int str_len = strlen(line);
        for(i = 0; i < str_len; ++i){
            c = line[i];
            input[(int)c] = 1;
            network_predict(net, input);
            input[(int)c] = 0;
        }
            c = ' ';
            input[(int)c] = 1;
            network_predict(net, input);
            input[(int)c] = 0;

        layer l = net.layers[0];
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
        printf("%s", line);
        for(i = 0; i < l.outputs; ++i){
            printf(",%g", l.output[i]);
        }
        printf("\n");
    }
}

void run_char_rnn(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *filename = find_char_arg(argc, argv, "-file", "data/shakespeare.txt");
    char *seed = find_char_arg(argc, argv, "-seed", "\n\n");
    int len = find_int_arg(argc, argv, "-len", 1000);
    float temp = find_float_arg(argc, argv, "-temp", .7);
    int rseed = find_int_arg(argc, argv, "-srand", time(0));
    int clear = find_arg(argc, argv, "-clear");

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_char_rnn(cfg, weights, filename, clear);
    else if(0==strcmp(argv[2], "valid")) valid_char_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "vec")) vec_char_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "generate")) test_char_rnn(cfg, weights, len, seed, temp, rseed);
}
