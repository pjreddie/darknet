#include "darknet.h"

#include <math.h>

typedef struct {
    float *x;
    float *y;
} float_pair;

unsigned char **load_files(char *filename, int *n)
{
    list *paths = get_paths(filename);
    *n = paths->size;
    unsigned char **contents = calloc(*n, sizeof(char *));
    int i;
    node *x = paths->front;
    for(i = 0; i < *n; ++i){
        contents[i] = read_file((char *)x->val);
        x = x->next;
    }
    return contents;
}

int *read_tokenized_data(char *filename, size_t *read)
{
    size_t size = 512;
    size_t count = 0;
    FILE *fp = fopen(filename, "r");
    int *d = calloc(size, sizeof(int));
    int n, one;
    one = fscanf(fp, "%d", &n);
    while(one == 1){
        ++count;
        if(count > size){
            size = size*2;
            d = realloc(d, size*sizeof(int));
        }
        d[count-1] = n;
        one = fscanf(fp, "%d", &n);
    }
    fclose(fp);
    d = realloc(d, count*sizeof(int));
    *read = count;
    return d;
}

char **read_tokens(char *filename, size_t *read)
{
    size_t size = 512;
    size_t count = 0;
    FILE *fp = fopen(filename, "r");
    char **d = calloc(size, sizeof(char *));
    char *line;
    while((line=fgetl(fp)) != 0){
        ++count;
        if(count > size){
            size = size*2;
            d = realloc(d, size*sizeof(char *));
        }
        if(0==strcmp(line, "<NEWLINE>")) line = "\n";
        d[count-1] = line;
    }
    fclose(fp);
    d = realloc(d, count*sizeof(char *));
    *read = count;
    return d;
}


float_pair get_rnn_token_data(int *tokens, size_t *offsets, int characters, size_t len, int batch, int steps)
{
    float *x = calloc(batch * steps * characters, sizeof(float));
    float *y = calloc(batch * steps * characters, sizeof(float));
    int i,j;
    for(i = 0; i < batch; ++i){
        for(j = 0; j < steps; ++j){
            int curr = tokens[(offsets[i])%len];
            int next = tokens[(offsets[i] + 1)%len];

            x[(j*batch + i)*characters + curr] = 1;
            y[(j*batch + i)*characters + next] = 1;

            offsets[i] = (offsets[i] + 1) % len;

            if(curr >= characters || curr < 0 || next >= characters || next < 0){
                error("Bad char");
            }
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}

float_pair get_seq2seq_data(char **source, char **dest, int n, int characters, size_t len, int batch, int steps)
{
    int i,j;
    float *x = calloc(batch * steps * characters, sizeof(float));
    float *y = calloc(batch * steps * characters, sizeof(float));
    for(i = 0; i < batch; ++i){
        int index = rand()%n;
        //int slen = strlen(source[index]);
        //int dlen = strlen(dest[index]);
        for(j = 0; j < steps; ++j){
            unsigned char curr = source[index][j];
            unsigned char next = dest[index][j];

            x[(j*batch + i)*characters + curr] = 1;
            y[(j*batch + i)*characters + next] = 1;

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

void train_char_rnn(char *cfgfile, char *weightfile, char *filename, int clear, int tokenized)
{
    srand(time(0));
    unsigned char *text = 0;
    int *tokens = 0;
    size_t size;
    if(tokenized){
        tokens = read_tokenized_data(filename, &size);
    } else {
        text = read_file(filename);
        size = strlen((const char*)text);
    }

    char *backup_directory = "/home/pjreddie/backup/";
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);
    float avg_loss = -1;
    network *net = load_network(cfgfile, weightfile, clear);

    int inputs = net->inputs;
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g, Inputs: %d %d %d\n", net->learning_rate, net->momentum, net->decay, inputs, net->batch, net->time_steps);
    int batch = net->batch;
    int steps = net->time_steps;
    if(clear) *net->seen = 0;
    int i = (*net->seen)/net->batch;

    int streams = batch/steps;
    size_t *offsets = calloc(streams, sizeof(size_t));
    int j;
    for(j = 0; j < streams; ++j){
        offsets[j] = rand_size_t()%size;
    }

    clock_t time;
    while(get_current_batch(net) < net->max_batches){
        i += 1;
        time=clock();
        float_pair p;
        if(tokenized){
            p = get_rnn_token_data(tokens, offsets, inputs, size, streams, steps);
        }else{
            p = get_rnn_data(text, offsets, inputs, size, streams, steps);
        }

        copy_cpu(net->inputs*net->batch, p.x, 1, net->input, 1);
        copy_cpu(net->truths*net->batch, p.y, 1, net->truth, 1);
        float loss = train_network_datum(net) / (batch);
        free(p.x);
        free(p.y);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        size_t chars = get_current_batch(net)*batch;
        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds, %f epochs\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), (float) chars/size);

        for(j = 0; j < streams; ++j){
            //printf("%d\n", j);
            if(rand()%64 == 0){
                //fprintf(stderr, "Reset\n");
                offsets[j] = rand_size_t()%size;
                reset_network_state(net, j);
            }
        }

        if(i%10000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_symbol(int n, char **tokens){
    if(tokens){
        printf("%s ", tokens[n]);
    } else {
        printf("%c", n);
    }
}

void test_char_rnn(char *cfgfile, char *weightfile, int num, char *seed, float temp, int rseed, char *token_file)
{
    char **tokens = 0;
    if(token_file){
        size_t n;
        tokens = read_tokens(token_file, &n);
    }

    srand(rseed);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int i, j;
    for(i = 0; i < net->n; ++i) net->layers[i].temperature = temp;
    int c = 0;
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
        input[c] = 1;
        network_predict(net, input);
        input[c] = 0;
        print_symbol(c, tokens);
    }
    if(len) c = seed[len-1];
    print_symbol(c, tokens);
    for(i = 0; i < num; ++i){
        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;
        for(j = 32; j < 127; ++j){
            //printf("%d %c %f\n",j, j, out[j]);
        }
        for(j = 0; j < inputs; ++j){
            if (out[j] < .0001) out[j] = 0;
        }
        c = sample_array(out, inputs);
        print_symbol(c, tokens);
    }
    printf("\n");
}

void test_tactic_rnn_multi(char *cfgfile, char *weightfile, int num, float temp, int rseed, char *token_file)
{
    char **tokens = 0;
    if(token_file){
        size_t n;
        tokens = read_tokens(token_file, &n);
    }

    srand(rseed);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int i, j;
    for(i = 0; i < net->n; ++i) net->layers[i].temperature = temp;
    int c = 0;
    float *input = calloc(inputs, sizeof(float));
    float *out = 0;

    while(1){
        reset_network_state(net, 0);
        while((c = getc(stdin)) != EOF && c != 0){
            input[c] = 1;
            out = network_predict(net, input);
            input[c] = 0;
        }
        for(i = 0; i < num; ++i){
            for(j = 0; j < inputs; ++j){
                if (out[j] < .0001) out[j] = 0;
            }
            int next = sample_array(out, inputs);
            if(c == '.' && next == '\n') break;
            c = next;
            print_symbol(c, tokens);

            input[c] = 1;
            out = network_predict(net, input);
            input[c] = 0;
        }
        printf("\n");
    }
}

void test_tactic_rnn(char *cfgfile, char *weightfile, int num, float temp, int rseed, char *token_file)
{
    char **tokens = 0;
    if(token_file){
        size_t n;
        tokens = read_tokens(token_file, &n);
    }

    srand(rseed);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int i, j;
    for(i = 0; i < net->n; ++i) net->layers[i].temperature = temp;
    int c = 0;
    float *input = calloc(inputs, sizeof(float));
    float *out = 0;

    while((c = getc(stdin)) != EOF){
        input[c] = 1;
        out = network_predict(net, input);
        input[c] = 0;
    }
    for(i = 0; i < num; ++i){
        for(j = 0; j < inputs; ++j){
            if (out[j] < .0001) out[j] = 0;
        }
        int next = sample_array(out, inputs);
        if(c == '.' && next == '\n') break;
        c = next;
        print_symbol(c, tokens);

        input[c] = 1;
        out = network_predict(net, input);
        input[c] = 0;
    }
    printf("\n");
}

void valid_tactic_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int count = 0;
    int words = 1;
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
    int in = 0;
    while(c != EOF){
        int next = getc(stdin);
        if(next == EOF) break;
        if(next < 0 || next >= 255) error("Out of range character");

        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;

        if(c == '.' && next == '\n') in = 0;
        if(!in) {
            if(c == '>' && next == '>'){
                in = 1;
                ++words;
            }
            c = next;
            continue;
        }
        ++count;
        sum += log(out[next])/log2;
        c = next;
        printf("%d %d Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, words, pow(2, -sum/count), pow(2, -sum/words));
    }
}

void valid_char_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int count = 0;
    int words = 1;
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
        if(next == ' ' || next == '\n' || next == '\t') ++words;
        input[c] = 1;
        float *out = network_predict(net, input);
        input[c] = 0;
        sum += log(out[next])/log2;
        c = next;
        printf("%d BPC: %4.4f   Perplexity: %4.4f    Word Perplexity: %4.4f\n", count, -sum/count, pow(2, -sum/count), pow(2, -sum/words));
    }
}

void vec_char_rnn(char *cfgfile, char *weightfile, char *seed)
{
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);
    int inputs = net->inputs;

    int c;
    int seed_len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));
    int i;
    char *line;
    while((line=fgetl(stdin)) != 0){
        reset_network_state(net, 0);
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

        layer l = net->layers[0];
        #ifdef GPU
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
        #endif
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
    int tokenized = find_arg(argc, argv, "-tokenized");
    char *tokens = find_char_arg(argc, argv, "-tokens", 0);

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_char_rnn(cfg, weights, filename, clear, tokenized);
    else if(0==strcmp(argv[2], "valid")) valid_char_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "validtactic")) valid_tactic_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "vec")) vec_char_rnn(cfg, weights, seed);
    else if(0==strcmp(argv[2], "generate")) test_char_rnn(cfg, weights, len, seed, temp, rseed, tokens);
    else if(0==strcmp(argv[2], "generatetactic")) test_tactic_rnn(cfg, weights, len, temp, rseed, tokens);
}
