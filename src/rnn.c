#include "network.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

typedef struct {
    float *x;
    float *y;
} float_pair;

float_pair get_rnn_data(unsigned char *text, int characters, int len, int batch, int steps)
{
    float *x = calloc(batch * steps * characters, sizeof(float));
    float *y = calloc(batch * steps * characters, sizeof(float));
    int i,j;
    for(i = 0; i < batch; ++i){
        int index = rand() %(len - steps - 1);
        /*
        int done = 1;
        while(!done){
            index = rand() %(len - steps - 1);
            while(index < len-steps-1 && text[index++] != '\n');
            if (index < len-steps-1) done = 1;
            }
         */
        for(j = 0; j < steps; ++j){
            x[(j*batch + i)*characters + text[index + j]] = 1;
            y[(j*batch + i)*characters + text[index + j + 1]] = 1;

            if(text[index+j] > 255 || text[index+j] <= 0 || text[index+j+1] > 255 || text[index+j+1] <= 0){
                text[index+j+2] = 0;
                printf("%d %d %d %d %d\n", index, j, len, (int)text[index+j], (int)text[index+j+1]);
                printf("%s", text+index);
                error("Bad char");
            }
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}

void train_char_rnn(char *cfgfile, char *weightfile, char *filename)
{
    FILE *fp = fopen(filename, "rb");

    fseek(fp, 0, SEEK_END); 
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET); 

    unsigned char *text = calloc(size+1, sizeof(char));
    fread(text, 1, size, fp);
    fclose(fp);

    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    data_seed = time(0);
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
    //*net.seen = 0;
    int i = (*net.seen)/net.batch;

    clock_t time;
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        float_pair p = get_rnn_data(text, inputs, size, batch/steps, steps);

        float loss = train_network_datum(net, p.x, p.y) / (batch);
        free(p.x);
        free(p.y);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time));
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
        float r = rand_uniform(0,1);
        float sum = 0;
        input[(int)c] = 1;
        float *out = network_predict(net, input);
        input[(int)c] = 0;
        for(j = 0; j < inputs; ++j){
            sum += out[j];
            if(sum > r) break;
        }
        c = j;
    }
    printf("\n");
}

void valid_char_rnn(char *cfgfile, char *weightfile)
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
    float *input = calloc(inputs, sizeof(float));
    float sum = 0;
    c = getc(stdin);
    float log2 = log(2);
    while(c != EOF){
        int next = getc(stdin);
        if(next == EOF) break;
        ++count;
        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;
        sum += log(out[next])/log2;
        c = next;
    }
    printf("Perplexity: %f\n", pow(2, -sum/count));
}


void run_char_rnn(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *filename = find_char_arg(argc, argv, "-file", "data/shakespeare.txt");
    char *seed = find_char_arg(argc, argv, "-seed", "\n");
    int len = find_int_arg(argc, argv, "-len", 1000);
    float temp = find_float_arg(argc, argv, "-temp", .7);
    int rseed = find_int_arg(argc, argv, "-srand", time(0));

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_char_rnn(cfg, weights, filename);
    else if(0==strcmp(argv[2], "valid")) valid_char_rnn(cfg, weights);
    else if(0==strcmp(argv[2], "generate")) test_char_rnn(cfg, weights, len, seed, temp, rseed);
}
