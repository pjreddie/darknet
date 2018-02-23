#include "darknet.h"

#include <assert.h>
#include <math.h>
#include <unistd.h>

int inverted = 1;
int noi = 1;
static const int nind = 10;
int legal_go(float *b, float *ko, int p, int r, int c);
int check_ko(float *x, float *ko);

typedef struct {
    char **data;
    int n;
} moves;

char *fgetgo(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 96;
    char *line = malloc(size*sizeof(char));
    if(size != fread(line, sizeof(char), size, fp)){
        free(line);
        return 0;
    }

    return line;
}

moves load_go_moves(char *filename)
{
    moves m;
    m.n = 128;
    m.data = calloc(128, sizeof(char*));
    FILE *fp = fopen(filename, "rb");
    int count = 0;
    char *line = 0;
    while ((line = fgetgo(fp))) {
        if (count >= m.n) {
            m.n *= 2;
            m.data = realloc(m.data, m.n*sizeof(char*));
        }
        m.data[count] = line;
        ++count;
    }
    printf("%d\n", count);
    m.n = count;
    m.data = realloc(m.data, count*sizeof(char*));
    return m;
}

void string_to_board(char *s, float *board)
{
    int i, j;
    memset(board, 0, 2*19*19*sizeof(float));
    int count = 0;
    for(i = 0; i < 91; ++i){
        char c = s[i];
        for(j = 0; j < 4; ++j){
            int me = (c >> (2*j)) & 1;
            int you = (c >> (2*j + 1)) & 1;
            if (me) board[count] = 1;
            else if (you) board[count + 19*19] = 1;
            ++count;
            if(count >= 19*19) break;
        }
    }
}

void board_to_string(char *s, float *board)
{
    int i, j;
    memset(s, 0, (19*19/4+1)*sizeof(char));
    int count = 0;
    for(i = 0; i < 91; ++i){
        for(j = 0; j < 4; ++j){
            int me = (board[count] == 1);
            int you = (board[count + 19*19] == 1);
            if (me) s[i] = s[i] | (1<<(2*j));
            if (you) s[i] = s[i] | (1<<(2*j + 1));
            ++count;
            if(count >= 19*19) break;
        }
    }
}

static int occupied(float *b, int i)
{
    if (b[i]) return 1;
    if (b[i+19*19]) return -1;
    return 0;
}

data random_go_moves(moves m, int n)
{
    data d = {0};
    d.X = make_matrix(n, 19*19*3);
    d.y = make_matrix(n, 19*19+2);
    int i, j;
    for(i = 0; i < n; ++i){
        float *board = d.X.vals[i];
        float *label = d.y.vals[i];
        char *b = m.data[rand()%m.n];
        int player = b[0] - '0';
        int result = b[1] - '0';
        int row = b[2];
        int col = b[3];
        string_to_board(b+4, board);
        if(player > 0) for(j = 0; j < 19*19; ++j) board[19*19*2 + j] = 1;
        label[19*19+1] = (player==result);
        if(row >= 19 || col >= 19){
            label[19*19] = 1;
        } else {
            label[col + 19*row] = 1;
            if(occupied(board, col + 19*row)) printf("hey\n");
        }

        int flip = rand()%2;
        int rotate = rand()%4;
        image in = float_to_image(19, 19, 3, board);
        image out = float_to_image(19, 19, 1, label);
        if(flip){
            flip_image(in);
            flip_image(out);
        }
        rotate_image_cw(in, rotate);
        rotate_image_cw(out, rotate);
    }
    return d;
}


void train_go(char *cfgfile, char *weightfile, char *filename, int *gpus, int ngpus, int clear)
{
    int i;
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    char *backup_directory = "/home/pjreddie/backup/";

    char buff[256];
    moves m = load_go_moves(filename);
    //moves m = load_go_moves("games.txt");

    int N = m.n;
    printf("Moves: %d\n", N);
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time=what_time_is_it_now();

        data train = random_go_moves(m, net->batch*net->subdivisions*ngpus);
        printf("Loaded: %lf seconds\n", what_time_is_it_now() - time);
        time=what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 10);
        }
#else
        loss = train_network(net, train);
#endif
        free_data(train);

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory,base, epoch);
            save_weights(net, buff);

        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%10000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%ld.backup",backup_directory,base,get_current_batch(net));
            save_weights(net, buff);
        }
    }
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free(base);
}

static void propagate_liberty(float *board, int *lib, int *visited, int row, int col, int side)
{
    if (row < 0 || row > 18 || col < 0 || col > 18) return;
    int index = row*19 + col;
    if (occupied(board,index) != side) return;
    if (visited[index]) return;
    visited[index] = 1;
    lib[index] += 1;
    propagate_liberty(board, lib, visited, row+1, col, side);
    propagate_liberty(board, lib, visited, row-1, col, side);
    propagate_liberty(board, lib, visited, row, col+1, side);
    propagate_liberty(board, lib, visited, row, col-1, side);
}


static int *calculate_liberties(float *board)
{
    int *lib = calloc(19*19, sizeof(int));
    int visited[19*19];
    int i, j;
    for(j = 0; j < 19; ++j){
        for(i = 0; i < 19; ++i){
            memset(visited, 0, 19*19*sizeof(int));
            int index = j*19 + i;
            if(!occupied(board,index)){
                if ((i > 0)  && occupied(board,index - 1)) propagate_liberty(board, lib, visited, j, i-1, occupied(board,index-1));
                if ((i < 18) && occupied(board,index + 1)) propagate_liberty(board, lib, visited, j, i+1, occupied(board,index+1));
                if ((j > 0)  && occupied(board,index - 19)) propagate_liberty(board, lib, visited, j-1, i, occupied(board,index-19));
                if ((j < 18) && occupied(board,index + 19)) propagate_liberty(board, lib, visited, j+1, i, occupied(board,index+19));
            }
        }
    }
    return lib;
}

void print_board(FILE *stream, float *board, int player, int *indexes)
{
    int i,j,n;
    fprintf(stream, "   ");
    for(i = 0; i < 19; ++i){
        fprintf(stream, "%c ", 'A' + i + 1*(i > 7 && noi));
    }
    fprintf(stream, "\n");
    for(j = 0; j < 19; ++j){
        fprintf(stream, "%2d", (inverted) ? 19-j : j+1);
        for(i = 0; i < 19; ++i){
            int index = j*19 + i;
            if(indexes){
                int found = 0;
                for(n = 0; n < nind; ++n){
                    if(index == indexes[n]){
                        found = 1;
                        /*
                           if(n == 0) fprintf(stream, "\uff11");
                           else if(n == 1) fprintf(stream, "\uff12");
                           else if(n == 2) fprintf(stream, "\uff13");
                           else if(n == 3) fprintf(stream, "\uff14");
                           else if(n == 4) fprintf(stream, "\uff15");
                         */
                        fprintf(stream, " %d", n+1);
                    }
                }
                if(found) continue;
            }
            //if(board[index]*-swap > 0) fprintf(stream, "\u25C9 ");
            //else if(board[index]*-swap < 0) fprintf(stream, "\u25EF ");
            if      (occupied(board, index) == player) fprintf(stream, " X");
            else if (occupied(board, index) ==-player) fprintf(stream, " O");
            else fprintf(stream, " .");
        }
        fprintf(stream, "\n");
    }
}

void flip_board(float *board)
{
    int i;
    for(i = 0; i < 19*19; ++i){
        float swap = board[i];
        board[i] = board[i+19*19];
        board[i+19*19] = swap;
        board[i+19*19*2] = 1-board[i+19*19*2];
    }
}

float predict_move2(network *net, float *board, float *move, int multi)
{
    float *output = network_predict(net, board);
    copy_cpu(19*19+1, output, 1, move, 1);
    float result = output[19*19 + 1];
    int i;
    if(multi){
        image bim = float_to_image(19, 19, 3, board);
        for(i = 1; i < 8; ++i){
            rotate_image_cw(bim, i);
            if(i >= 4) flip_image(bim);

            float *output = network_predict(net, board);
            image oim = float_to_image(19, 19, 1, output);
            result += output[19*19 + 1];

            if(i >= 4) flip_image(oim);
            rotate_image_cw(oim, -i);

            axpy_cpu(19*19+1, 1, output, 1, move, 1);

            if(i >= 4) flip_image(bim);
            rotate_image_cw(bim, -i);
        }
        result = result/8;
        scal_cpu(19*19+1, 1./8., move, 1);
    }
    for(i = 0; i < 19*19; ++i){
        if(board[i] || board[i+19*19]) move[i] = 0;
    }
    return result;
}

static void remove_connected(float *b, int *lib, int p, int r, int c)
{
    if (r < 0 || r >= 19 || c < 0 || c >= 19) return;
    if (occupied(b, r*19 + c) != p) return;
    if (lib[r*19 + c] != 1) return;
    b[r*19 + c] = 0;
    b[19*19 + r*19 + c] = 0;
    remove_connected(b, lib, p, r+1, c);
    remove_connected(b, lib, p, r-1, c);
    remove_connected(b, lib, p, r, c+1);
    remove_connected(b, lib, p, r, c-1);
}


void move_go(float *b, int p, int r, int c)
{
    int *l = calculate_liberties(b);
    if(p > 0) b[r*19 + c] = 1;
    else b[19*19 + r*19 + c] = 1;
    remove_connected(b, l, -p, r+1, c);
    remove_connected(b, l, -p, r-1, c);
    remove_connected(b, l, -p, r, c+1);
    remove_connected(b, l, -p, r, c-1);
    free(l);
}

int compare_board(float *a, float *b)
{
    if(memcmp(a, b, 19*19*3*sizeof(float)) == 0) return 1;
    return 0;
}

typedef struct mcts_tree{
    float *board;
    struct mcts_tree **children;
    float *prior;
    int *visit_count;
    float *value;
    float *mean;
    float *prob;
    int total_count;
    float result;
    int done;
    int pass;
} mcts_tree;

void free_mcts(mcts_tree *root)
{
    if(!root) return;
    int i;
    free(root->board);
    for(i = 0; i < 19*19+1; ++i){
        if(root->children[i]) free_mcts(root->children[i]);
    }
    free(root->children);
    free(root->prior);
    free(root->visit_count);
    free(root->value);
    free(root->mean);
    free(root->prob);
    free(root);
}

float *network_predict_rotations(network *net, float *next)
{
    int n = net->batch;
    float *in = calloc(19*19*3*n, sizeof(float));
    image im = float_to_image(19, 19, 3, next);
    int i,j;
    int *inds = random_index_order(0, 8);
    for(j = 0; j < n; ++j){
        i = inds[j];
        rotate_image_cw(im, i);
        if(i >= 4) flip_image(im);
        memcpy(in + 19*19*3*j, im.data, 19*19*3*sizeof(float));
        if(i >= 4) flip_image(im);
        rotate_image_cw(im, -i);
    }
    float *pred = network_predict(net, in);
    for(j = 0; j < n; ++j){
        i = inds[j];
        image im = float_to_image(19, 19, 1, pred + j*(19*19 + 2));
        if(i >= 4) flip_image(im);
        rotate_image_cw(im, -i);
        if(j > 0){
            axpy_cpu(19*19+2, 1, im.data, 1, pred, 1);
        }
    }
    free(in);
    free(inds);
    scal_cpu(19*19+2, 1./n, pred, 1);
    return pred;
}

mcts_tree *expand(float *next, float *ko, network *net)
{
    mcts_tree *root = calloc(1, sizeof(mcts_tree));
    root->board = next;
    root->children = calloc(19*19+1, sizeof(mcts_tree*));
    root->prior = calloc(19*19 + 1, sizeof(float));
    root->prob = calloc(19*19 + 1, sizeof(float));
    root->mean = calloc(19*19 + 1, sizeof(float));
    root->value = calloc(19*19 + 1, sizeof(float));
    root->visit_count = calloc(19*19 + 1, sizeof(int));
    root->total_count = 1;
    int i;
    float *pred = network_predict_rotations(net, next);
    copy_cpu(19*19+1, pred, 1, root->prior, 1);
    float val = 2*pred[19*19 + 1] - 1;
    root->result = val;
    for(i = 0; i < 19*19+1; ++i) {
        root->visit_count[i] = 0;
        root->value[i] = 0;
        root->mean[i] = val;
        if(i < 19*19 && occupied(next, i)){
            root->value[i] = -1;
            root->mean[i] = -1;
            root->prior[i] = 0;
        }
    }
    //print_board(stderr, next, flip?-1:1, 0);
    return root;
}

float *copy_board(float *board)
{
    float *next = calloc(19*19*3, sizeof(float));
    copy_cpu(19*19*3, board, 1, next, 1);
    return next;
}

float select_mcts(mcts_tree *root, network *net, float *prev, float cpuct)
{
    if(root->done) return -root->result;
    int i;
    float max = -1000;
    int max_i = 0;
    for(i = 0; i < 19*19+1; ++i){
        root->prob[i] = root->mean[i] + cpuct*root->prior[i] * sqrt(root->total_count) / (1. + root->visit_count[i]);
        if(root->prob[i] > max){
            max = root->prob[i];
            max_i = i;
        }
    }
    float val;
    i = max_i;
    root->visit_count[i]++;
    root->total_count++;
    if (root->children[i]) {
        val = select_mcts(root->children[i], net, root->board, cpuct);
    } else {
        if(max_i < 19*19 && !legal_go(root->board, prev, 1, max_i/19, max_i%19)) {
            root->mean[i]  = -1;
            root->value[i] = -1;
            root->prior[i] = 0;
            --root->total_count;
            return select_mcts(root, net, prev, cpuct);
            //printf("Detected ko\n");
            //getchar();
        } else {
            float *next = copy_board(root->board);
            if (max_i < 19*19) {
                move_go(next, 1, max_i / 19, max_i % 19);
            }
            flip_board(next);
            root->children[i] = expand(next, root->board, net);
            val = -root->children[i]->result;
            if(max_i == 19*19){
                root->children[i]->pass = 1;
                if (root->pass){
                    root->children[i]->done = 1;
                }
            }
        }
    }
    root->value[i] += val;
    root->mean[i] = root->value[i]/root->visit_count[i];
    return -val;
}

mcts_tree *run_mcts(mcts_tree *tree, network *net, float *board, float *ko, int player, int n, float cpuct, float secs)
{
    int i;
    double t = what_time_is_it_now();
    if(player < 0) flip_board(board);
    if(!tree) tree = expand(copy_board(board), ko, net);
    assert(compare_board(tree->board, board));
    for(i = 0; i < n; ++i){
        if (secs > 0 && (what_time_is_it_now() - t) > secs) break;
        int max_i = max_int_index(tree->visit_count, 19*19+1);
        if (tree->visit_count[max_i] >= n) break;
        select_mcts(tree, net, ko, cpuct);
    }
    if(player < 0) flip_board(board);
    //fprintf(stderr, "%f Seconds\n", what_time_is_it_now() - t);
    return tree;
}

mcts_tree *move_mcts(mcts_tree *tree, int index)
{
    if(index < 0 || index > 19*19 || !tree || !tree->children[index]) {
        free_mcts(tree);
        tree = 0;
    } else {
        mcts_tree *swap = tree;
        tree = tree->children[index];
        swap->children[index] = 0;
        free_mcts(swap);
    }
    return tree;
}

typedef struct {
    float value;
    float mcts;
    int row;
    int col;
} move;

move pick_move(mcts_tree *tree, float temp, int player)
{
    int i;
    float probs[19*19+1] = {0};
    move m = {0};
    double sum = 0;
    /*
    for(i = 0; i < 19*19+1; ++i){
        probs[i] = tree->visit_count[i];
    }
    */
    //softmax(probs, 19*19+1, temp, 1, probs);
    for(i = 0; i < 19*19+1; ++i){
        sum += pow(tree->visit_count[i], 1./temp);
    }
    for(i = 0; i < 19*19+1; ++i){
        probs[i] = pow(tree->visit_count[i], 1./temp) / sum;
    }

    int index = sample_array(probs, 19*19+1);
    m.row = index / 19;
    m.col = index % 19;
    m.value = (tree->result+1.)/2.;
    m.mcts  = (tree->mean[index]+1.)/2.;

    int indexes[nind];
    top_k(probs, 19*19+1, nind, indexes);
    print_board(stderr, tree->board, player, indexes);

    fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", index/19, index%19, tree->result, tree->prior[index], probs[index], tree->mean[index], (tree->children[index])?tree->children[index]->result:0, tree->visit_count[index]);
    int ind = max_index(probs, 19*19+1);
    fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", ind/19, ind%19, tree->result, tree->prior[ind], probs[ind], tree->mean[ind], (tree->children[ind])?tree->children[ind]->result:0, tree->visit_count[ind]);
    ind = max_index(tree->prior, 19*19+1);
    fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", ind/19, ind%19, tree->result, tree->prior[ind], probs[ind], tree->mean[ind], (tree->children[ind])?tree->children[ind]->result:0, tree->visit_count[ind]);
    return m;
}

/*
   float predict_move(network *net, float *board, float *move, int multi, float *ko, float temp)
   {

   int i;

   int max_v = 0;
   int max_i = 0;
   for(i = 0; i < 19*19+1; ++i){
   if(root->visit_count[i] > max_v){
   max_v = root->visit_count[i];
   max_i = i;
   }
   }
   fprintf(stderr, "%f Seconds\n", what_time_is_it_now() - t);
   int ind = max_index(root->mean, 19*19+1);
   fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", max_i/19, max_i%19, root->result, root->prior[max_i], root->prob[max_i], root->mean[max_i], (root->children[max_i])?root->children[max_i]->result:0, root->visit_count[max_i]);
   fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", ind/19, ind%19, root->result, root->prior[ind], root->prob[ind], root->mean[ind], (root->children[ind])?root->children[ind]->result:0, root->visit_count[ind]);
   ind = max_index(root->prior, 19*19+1);
   fprintf(stderr, "%d %d, Result: %f, Prior: %f, Prob: %f, Mean Value: %f, Child Result: %f, Visited: %d\n", ind/19, ind%19, root->result, root->prior[ind], root->prob[ind], root->mean[ind], (root->children[ind])?root->children[ind]->result:0, root->visit_count[ind]);
   if(root->result < -.9 && root->mean[max_i] < -.9) return -1000.f;

   float val = root->result;
   free_mcts(root);
   return val;
   }
 */

static int makes_safe_go(float *b, int *lib, int p, int r, int c){
    if (r < 0 || r >= 19 || c < 0 || c >= 19) return 0;
    if (occupied(b,r*19 + c) == -p){
        if (lib[r*19 + c] > 1) return 0;
        else return 1;
    }
    if (!occupied(b,r*19 + c)) return 1;
    if (lib[r*19 + c] > 1) return 1;
    return 0;
}

int suicide_go(float *b, int p, int r, int c)
{
    int *l = calculate_liberties(b);
    int safe = 0;
    safe = safe || makes_safe_go(b, l, p, r+1, c);
    safe = safe || makes_safe_go(b, l, p, r-1, c);
    safe = safe || makes_safe_go(b, l, p, r, c+1);
    safe = safe || makes_safe_go(b, l, p, r, c-1);
    free(l);
    return !safe;
}

int check_ko(float *x, float *ko)
{
    if(!ko) return 0;
    float curr[19*19*3];
    copy_cpu(19*19*3, x, 1, curr, 1);
    if(curr[19*19*2] != ko[19*19*2]) flip_board(curr);
    if(compare_board(curr, ko)) return 1;
    return 0;
}

int legal_go(float *b, float *ko, int p, int r, int c)
{
    if (occupied(b, r*19+c)) return 0;
    float curr[19*19*3];
    copy_cpu(19*19*3, b, 1, curr, 1);
    move_go(curr, p, r, c);
    if(check_ko(curr, ko)) return 0;
    if(suicide_go(b, p, r, c)) return 0;
    return 1;
}

/*
   move generate_move(mcts_tree *root, network *net, int player, float *board, int multi, float temp, float *ko, int print)
   {
   move m = {0};
//root = run_mcts(tree, network *net, float *board, float *ko, int n, float cpuct)
int i, j;
int empty = 1;
for(i = 0; i < 19*19; ++i){
if (occupied(board, i)) {
empty = 0;
break;
}
}
if(empty) {
m.value = .5;
m.mcts = .5;
m.row = 3;
m.col = 15;
return m;
}

float move[362];
if (player < 0) flip_board(board);
float result = predict_move(net, board, move, multi, ko, temp);
if (player < 0) flip_board(board);
if(result == -1000.f) return -2;

for(i = 0; i < 19; ++i){
for(j = 0; j < 19; ++j){
if (!legal_go(board, ko, player, i, j)) move[i*19 + j] = 0;
}
}

int indexes[nind];
top_k(move, 19*19+1, nind, indexes);


int max = max_index(move, 19*19+1);
int row = max / 19;
int col = max % 19;
int index = sample_array(move, 19*19+1);

if(print){
top_k(move, 19*19+1, nind, indexes);
for(i = 0; i < nind; ++i){
if (!move[indexes[i]]) indexes[i] = -1;
}
print_board(stderr, board, 1, indexes);
fprintf(stderr, "%s To Move\n", player > 0 ? "X" : "O");
fprintf(stderr, "%.2f%% Win Chance\n", (result+1)/2*100);
for(i = 0; i < nind; ++i){
int index = indexes[i];
int row = index / 19;
int col = index % 19;
if(row == 19){
fprintf(stderr, "%d: Pass, %.2f%%\n", i+1, move[index]*100);
} else {
fprintf(stderr, "%d: %c %d, %.2f%%\n", i+1, col + 'A' + 1*(col > 7 && noi), (inverted)?19 - row : row+1, move[index]*100);
}
}
}
if (row == 19) return -1;

if (suicide_go(board, player, row, col)){
return -1; 
}

if (suicide_go(board, player, index/19, index%19)){
index = max;
}
if (index == 19*19) return -1;
return index;
}
*/

void valid_go(char *cfgfile, char *weightfile, int multi, char *filename)
{
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    float *board = calloc(19*19*3, sizeof(float));
    float *move = calloc(19*19+2, sizeof(float));
    // moves m = load_go_moves("/home/pjreddie/backup/go.test");
    moves m = load_go_moves(filename);

    int N = m.n;
    int i,j;
    int correct = 0;
    for (i = 0; i <N; ++i) {
        char *b = m.data[i];
        int player = b[0] - '0';
        //int result = b[1] - '0';
        int row = b[2];
        int col = b[3];
        int truth = col + 19*row;
        string_to_board(b+4, board);
        if(player > 0) for(j = 0; j < 19*19; ++j) board[19*19*2 + j] = 1;
        predict_move2(net, board, move, multi);
        int index = max_index(move, 19*19+1);
        if(index == truth) ++correct;
        printf("%d Accuracy %f\n", i, (float) correct/(i+1));
    }
}

int print_game(float *board, FILE *fp)
{
    int i, j;
    int count = 3;
    fprintf(fp, "komi 6.5\n");
    fprintf(fp, "boardsize 19\n");
    fprintf(fp, "clear_board\n");
    for(j = 0; j < 19; ++j){
        for(i = 0; i < 19; ++i){
            if(occupied(board,j*19 + i) == 1) fprintf(fp, "play black %c%d\n", 'A'+i+(i>=8), 19-j);
            if(occupied(board,j*19 + i) == -1) fprintf(fp, "play white %c%d\n", 'A'+i+(i>=8), 19-j);
            if(occupied(board,j*19 + i)) ++count;
        }
    }
    return count;
}


int stdin_ready()
{
    fd_set readfds;
    FD_ZERO(&readfds);

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;
    FD_SET(STDIN_FILENO, &readfds);

    if (select(1, &readfds, NULL, NULL, &timeout)){
        return 1;
    }
    return 0;
}

mcts_tree *ponder(mcts_tree *tree, network *net, float *b, float *ko, int player, float cpuct)
{
    double t = what_time_is_it_now();
    int count = 0;
    if (tree) count = tree->total_count;
    while(!stdin_ready()){
        if (what_time_is_it_now() - t > 120) break;
        tree = run_mcts(tree, net, b, ko, player, 100000, cpuct, .1);
    }
    fprintf(stderr, "Pondered %d moves...\n", tree->total_count - count);
    return tree;
}

void engine_go(char *filename, char *weightfile, int mcts_iters, float secs, float temp, float cpuct, int anon, int resign)
{
    mcts_tree *root = 0;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));
    float *board = calloc(19*19*3, sizeof(float));
    flip_board(board);
    float *one = calloc(19*19*3, sizeof(float));
    float *two = calloc(19*19*3, sizeof(float));
    int ponder_player = 0;
    int passed = 0;
    int move_num = 0;
    int main_time = 0;
    int byo_yomi_time = 0;
    int byo_yomi_stones = 0;
    int black_time_left = 0;
    int black_stones_left = 0;
    int white_time_left = 0;
    int white_stones_left = 0;
    float orig_time = secs;
    int old_ponder = 0;
    while(1){
        if(ponder_player){
            root = ponder(root, net, board, two, ponder_player, cpuct);
        }
        old_ponder = ponder_player;
        ponder_player = 0;
        char buff[256];
        int id = 0;
        int has_id = (scanf("%d", &id) == 1);
        scanf("%s", buff);
        if (feof(stdin)) break;
        fprintf(stderr, "%s\n", buff);
        char ids[256];
        sprintf(ids, "%d", id);
        //fprintf(stderr, "%s\n", buff);
        if (!has_id) ids[0] = 0;
        if (!strcmp(buff, "protocol_version")){
            printf("=%s 2\n\n", ids);
        } else if (!strcmp(buff, "name")){
            if(anon){
                printf("=%s The Fool!\n\n", ids);
            }else{
                printf("=%s DarkGo\n\n", ids);
            }
        } else if (!strcmp(buff, "time_settings")){
            ponder_player = old_ponder;
            scanf("%d %d %d", &main_time, &byo_yomi_time, &byo_yomi_stones);
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "time_left")){
            ponder_player = old_ponder;
            char color[256];
            int time = 0, stones = 0;
            scanf("%s %d %d", color, &time, &stones);
            if (color[0] == 'b' || color[0] == 'B'){
                black_time_left = time;
                black_stones_left = stones;
            } else {
                white_time_left = time;
                white_stones_left = stones;
            }
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "version")){
            if(anon){
                printf("=%s :-DDDD\n\n", ids);
            }else {
                printf("=%s 1.0. Want more DarkGo? You can find me on OGS, unlimited games, no waiting! https://online-go.com/user/view/434218\n\n", ids);
            }
        } else if (!strcmp(buff, "known_command")){
            char comm[256];
            scanf("%s", comm);
            int known = (!strcmp(comm, "protocol_version") || 
                    !strcmp(comm, "name") || 
                    !strcmp(comm, "version") || 
                    !strcmp(comm, "known_command") || 
                    !strcmp(comm, "list_commands") || 
                    !strcmp(comm, "quit") || 
                    !strcmp(comm, "boardsize") || 
                    !strcmp(comm, "clear_board") || 
                    !strcmp(comm, "komi") || 
                    !strcmp(comm, "final_status_list") || 
                    !strcmp(comm, "play") || 
                    !strcmp(comm, "genmove_white") || 
                    !strcmp(comm, "genmove_black") || 
                    !strcmp(comm, "fixed_handicap") || 
                    !strcmp(comm, "genmove"));
            if(known) printf("=%s true\n\n", ids);
            else printf("=%s false\n\n", ids);
        } else if (!strcmp(buff, "list_commands")){
            printf("=%s protocol_version\nshowboard\nname\nversion\nknown_command\nlist_commands\nquit\nboardsize\nclear_board\nkomi\nplay\ngenmove_black\ngenmove_white\ngenmove\nfinal_status_list\nfixed_handicap\n\n", ids);
        } else if (!strcmp(buff, "quit")){
            break;
        } else if (!strcmp(buff, "boardsize")){
            int boardsize = 0;
            scanf("%d", &boardsize);
            //fprintf(stderr, "%d\n", boardsize);
            if(boardsize != 19){
                printf("?%s unacceptable size\n\n", ids);
            } else {
                root = move_mcts(root, -1);
                memset(board, 0, 3*19*19*sizeof(float));
                flip_board(board);
                move_num = 0;
                printf("=%s \n\n", ids);
            }
        } else if (!strcmp(buff, "fixed_handicap")){
            int handicap = 0;
            scanf("%d", &handicap);
            int indexes[] = {72, 288, 300, 60, 180, 174, 186, 66, 294};
            int i;
            for(i = 0; i < handicap; ++i){
                board[indexes[i]] = 1;   
                ++move_num;
            }
            root = move_mcts(root, -1);
        } else if (!strcmp(buff, "clear_board")){
            passed = 0;
            memset(board, 0, 3*19*19*sizeof(float));
            flip_board(board);
            move_num = 0;
            root = move_mcts(root, -1);
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "komi")){
            float komi = 0;
            scanf("%f", &komi);
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "showboard")){
            printf("=%s \n", ids);
            print_board(stdout, board, 1, 0);
            printf("\n");
        } else if (!strcmp(buff, "play") || !strcmp(buff, "black") || !strcmp(buff, "white")){
            ++move_num;
            char color[256];
            if(!strcmp(buff, "play"))
            {
                scanf("%s ", color);
            } else {
                scanf(" ");
                color[0] = buff[0];
            }
            char c;
            int r;
            int count = scanf("%c%d", &c, &r);
            int player = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
            if((c == 'p' || c == 'P') && count < 2) {
                passed = 1;
                printf("=%s \n\n", ids);
                char *line = fgetl(stdin);
                free(line);
                fflush(stdout);
                fflush(stderr);
                root = move_mcts(root, 19*19);
                continue;
            } else {
                passed = 0;
            }
            if(c >= 'A' && c <= 'Z') c = c - 'A';
            if(c >= 'a' && c <= 'z') c = c - 'a';
            if(c >= 8) --c;
            r = 19 - r;
            fprintf(stderr, "move: %d %d\n", r, c);

            float *swap = two;
            two = one;
            one = swap;
            move_go(board, player, r, c);
            copy_cpu(19*19*3, board, 1, one, 1);
            if(root) fprintf(stderr, "Prior: %f\n", root->prior[r*19 + c]);
            if(root) fprintf(stderr, "Mean: %f\n", root->mean[r*19 + c]);
            if(root) fprintf(stderr, "Result: %f\n", root->result);
            root = move_mcts(root, r*19 + c);
            if(root) fprintf(stderr, "Visited: %d\n", root->total_count);
            else fprintf(stderr, "NOT VISITED\n");

            printf("=%s \n\n", ids);
            //print_board(stderr, board, 1, 0);
        } else if (!strcmp(buff, "genmove") || !strcmp(buff, "genmove_black") || !strcmp(buff, "genmove_white")){
            ++move_num;
            int player = 0;
            if(!strcmp(buff, "genmove")){
                char color[256];
                scanf("%s", color);
                player = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
            } else if (!strcmp(buff, "genmove_black")){
                player = 1;
            } else {
                player = -1;
            }
            if(player > 0){
                if(black_time_left <= 30) secs = 2.5;
                else secs = orig_time;
            } else {
                if(white_time_left <= 30) secs = 2.5;
                else secs = orig_time;
            }
            ponder_player = -player;

            //tree = generate_move(net, player, board, multi, .1, two, 1);
            double t = what_time_is_it_now();
            root = run_mcts(root, net, board, two, player, mcts_iters, cpuct, secs);
            fprintf(stderr, "%f Seconds\n", what_time_is_it_now() - t);
            move m = pick_move(root, temp, player);
            root = move_mcts(root, m.row*19 + m.col);


            if(move_num > resign && m.value < .1 && m.mcts < .1){
                printf("=%s resign\n\n", ids);
            } else if(m.row == 19){
                printf("=%s pass\n\n", ids);
                passed = 0;
            } else {
                int row = m.row;
                int col = m.col;

                float *swap = two;
                two = one;
                one = swap;

                move_go(board, player, row, col);
                copy_cpu(19*19*3, board, 1, one, 1);
                row = 19 - row;
                if (col >= 8) ++col;
                printf("=%s %c%d\n\n", ids, 'A' + col, row);
            }

        } else if (!strcmp(buff, "p")){
            //print_board(board, 1, 0);
        } else if (!strcmp(buff, "final_status_list")){
            char type[256];
            scanf("%s", type);
            fprintf(stderr, "final_status\n");
            char *line = fgetl(stdin);
            free(line);
            if(type[0] == 'd' || type[0] == 'D'){
                int i;
                FILE *f = fopen("game.txt", "w");
                int count = print_game(board, f);
                fprintf(f, "%s final_status_list dead\n", ids);
                fclose(f);
                FILE *p = popen("./gnugo --mode gtp < game.txt", "r");
                for(i = 0; i < count; ++i){
                    free(fgetl(p));
                    free(fgetl(p));
                }
                char *l = 0;
                while((l = fgetl(p))){
                    printf("%s\n", l);
                    free(l);
                }
            } else {
                printf("?%s unknown command\n\n", ids);
            }
        } else if (!strcmp(buff, "kgs-genmove_cleanup")){
            char type[256];
            scanf("%s", type);
            fprintf(stderr, "kgs-genmove_cleanup\n");
            char *line = fgetl(stdin);
            free(line);
            int i;
            FILE *f = fopen("game.txt", "w");
            int count = print_game(board, f);
            fprintf(f, "%s kgs-genmove_cleanup %s\n", ids, type);
            fclose(f);
            FILE *p = popen("./gnugo --mode gtp < game.txt", "r");
            for(i = 0; i < count; ++i){
                free(fgetl(p));
                free(fgetl(p));
            }
            char *l = 0;
            while((l = fgetl(p))){
                printf("%s\n", l);
                free(l);
            }
        } else {
            char *line = fgetl(stdin);
            free(line);
            printf("?%s unknown command\n\n", ids);
        }
        fflush(stdout);
        fflush(stderr);
    }
    printf("%d %d %d\n",passed, black_stones_left, white_stones_left);
}

void test_go(char *cfg, char *weights, int multi)
{
    int i;
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(time(0));
    float *board = calloc(19*19*3, sizeof(float));
    flip_board(board);
    float *move = calloc(19*19+1, sizeof(float));
    int color = 1;
    while(1){
        float result = predict_move2(net, board, move, multi);
        printf("%.2f%% Win Chance\n", (result+1)/2*100);

        int indexes[nind];
        int row, col;
        top_k(move, 19*19+1, nind, indexes);
        print_board(stderr, board, color, indexes);
        for(i = 0; i < nind; ++i){
            int index = indexes[i];
            row = index / 19;
            col = index % 19;
            if(row == 19){
                printf("%d: Pass, %.2f%%\n", i+1, move[index]*100);
            } else {
                printf("%d: %c %d, %.2f%%\n", i+1, col + 'A' + 1*(col > 7 && noi), (inverted)?19 - row : row+1, move[index]*100);
            }
        }
        //if(color == 1) printf("\u25EF Enter move: ");
        //else printf("\u25C9 Enter move: ");
        if(color == 1) printf("X Enter move: ");
        else printf("O Enter move: ");

        char c;
        char *line = fgetl(stdin);
        int picked = 1;
        int dnum = sscanf(line, "%d", &picked);
        int cnum = sscanf(line, "%c", &c);
        if (strlen(line) == 0 || dnum) {
            --picked;
            if (picked < nind){
                int index = indexes[picked];
                row = index / 19;
                col = index % 19;
                if(row < 19){
                    move_go(board, 1, row, col);
                }
            }
        } else if (cnum){
            if (c <= 'T' && c >= 'A'){
                int num = sscanf(line, "%c %d", &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 2) move_go(board, 1, row, col);
            } else if (c == 'p') {
                // Pass
            } else if(c=='b' || c == 'w'){
                char g;
                int num = sscanf(line, "%c %c %d", &g, &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 3) {
                    int mc = (g == 'b') ? 1 : -1;
                    if (mc == color) {
                        board[row*19 + col] = 1;
                    } else {
                        board[19*19 + row*19 + col] = 1;
                    }
                }
            } else if(c == 'c'){
                char g;
                int num = sscanf(line, "%c %c %d", &g, &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 3) {
                    board[row*19 + col] = 0;
                    board[19*19 + row*19 + col] = 0;
                }
            }
        }
        free(line);
        flip_board(board);
        color = -color;
    }
}

float score_game(float *board)
{
    int i;
    FILE *f = fopen("game.txt", "w");
    int count = print_game(board, f);
    fprintf(f, "final_score\n");
    fclose(f);
    FILE *p = popen("./gnugo --mode gtp < game.txt", "r");
    for(i = 0; i < count; ++i){
        free(fgetl(p));
        free(fgetl(p));
    }
    char *l = 0;
    float score = 0;
    char player = 0;
    while((l = fgetl(p))){
        fprintf(stderr, "%s  \t", l);
        int n = sscanf(l, "= %c+%f", &player, &score);
        free(l);
        if (n == 2) break;
    }
    if(player == 'W') score = -score;
    pclose(p);
    return score;
}

void self_go(char *filename, char *weightfile, char *f2, char *w2, int multi)
{
    mcts_tree *tree1 = 0;
    mcts_tree *tree2 = 0;
    network *net = load_network(filename, weightfile, 0);
    //set_batch_network(net, 1);

    network *net2;
    if (f2) {
        net2 = parse_network_cfg(f2);
        if(w2){
            load_weights(net2, w2);
        }
    } else {
        net2 = calloc(1, sizeof(network));
        *net2 = *net;
    }
    srand(time(0));
    char boards[600][93];
    int count = 0;
    //set_batch_network(net, 1);
    //set_batch_network(net2, 1);
    float *board = calloc(19*19*3, sizeof(float));
    flip_board(board);
    float *one = calloc(19*19*3, sizeof(float));
    float *two = calloc(19*19*3, sizeof(float));
    int done = 0;
    int player = 1;
    int p1 = 0;
    int p2 = 0;
    int total = 0;
    float temp = .1;
    int mcts_iters = 500;
    float cpuct = 5;
    while(1){
        if (done){
            tree1 = move_mcts(tree1, -1);
            tree2 = move_mcts(tree2, -1);
            float score = score_game(board);
            if((score > 0) == (total%2==0)) ++p1;
            else ++p2;
            ++total;
            fprintf(stderr, "Total: %d, Player 1: %f, Player 2: %f\n", total, (float)p1/total, (float)p2/total);
            sleep(1);
            /*
               int i = (score > 0)? 0 : 1;
               int j;
               for(; i < count; i += 2){
               for(j = 0; j < 93; ++j){
               printf("%c", boards[i][j]);
               }
               printf("\n");
               }
             */
            memset(board, 0, 3*19*19*sizeof(float));
            flip_board(board);
            player = 1;
            done = 0;
            count = 0;
            fflush(stdout);
            fflush(stderr);
        }
        //print_board(stderr, board, 1, 0);
        //sleep(1);

        if ((total%2==0) == (player==1)){
            //mcts_iters = 4500;   
            cpuct = 5;
        } else {
            //mcts_iters = 500;
            cpuct = 1;
        }
        network *use = ((total%2==0) == (player==1)) ? net : net2;
        mcts_tree *t = ((total%2==0) == (player==1)) ? tree1 : tree2;
        t = run_mcts(t, use, board, two, player, mcts_iters, cpuct, 0);
        move m = pick_move(t, temp, player);
        if(((total%2==0) == (player==1))) tree1 = t;
        else tree2 = t;

        tree1 = move_mcts(tree1, m.row*19 + m.col);
        tree2 = move_mcts(tree2, m.row*19 + m.col);

        if(m.row == 19){
            done = 1;
            continue;
        }
        int row = m.row;
        int col = m.col;

        float *swap = two;
        two = one;
        one = swap;

        if(player < 0) flip_board(board);
        boards[count][0] = row;
        boards[count][1] = col;
        board_to_string(boards[count] + 2, board);
        if(player < 0) flip_board(board);
        ++count;

        move_go(board, player, row, col);
        copy_cpu(19*19*3, board, 1, one, 1);

        player = -player;
    }
}

void run_go(int argc, char **argv)
{
    //boards_go();
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
    int clear = find_arg(argc, argv, "-clear");

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *c2 = (argc > 5) ? argv[5] : 0;
    char *w2 = (argc > 6) ? argv[6] : 0;
    int multi = find_arg(argc, argv, "-multi");
    int anon = find_arg(argc, argv, "-anon");
    int iters = find_int_arg(argc, argv, "-iters", 500);
    int resign = find_int_arg(argc, argv, "-resign", 175);
    float cpuct = find_float_arg(argc, argv, "-cpuct", 5);
    float temp = find_float_arg(argc, argv, "-temp", .1);
    float time = find_float_arg(argc, argv, "-time", 0);
    if(0==strcmp(argv[2], "train")) train_go(cfg, weights, c2, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) valid_go(cfg, weights, multi, c2);
    else if(0==strcmp(argv[2], "self")) self_go(cfg, weights, c2, w2, multi);
    else if(0==strcmp(argv[2], "test")) test_go(cfg, weights, multi);
    else if(0==strcmp(argv[2], "engine")) engine_go(cfg, weights, iters, time, temp, cpuct, anon, resign);
}


