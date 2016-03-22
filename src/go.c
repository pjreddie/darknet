#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

int inverted = 1;
int noi = 1;
static const int nind = 5;

void train_go(char *cfgfile, char *weightfile)
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

    char *backup_directory = "/home/pjreddie/backup/";


    char buff[256];
    sprintf(buff, "/home/pjreddie/go.train.%02d", rand()%10);
    data train = load_go(buff);

    int N = train.X.rows;
    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        clock_t time=clock();

        data batch = get_random_data(train, net.batch);
        int i;
        for(i = 0; i < batch.X.rows; ++i){
            int flip = rand()%2;
            int rotate = rand()%4;
            image in = float_to_image(19, 19, 1, batch.X.vals[i]);
            image out = float_to_image(19, 19, 1, batch.y.vals[i]);
            //show_image_normalized(in, "in");
            //show_image_normalized(out, "out");
            if(flip){
                flip_image(in);
                flip_image(out);
            }
            rotate_image_cw(in, rotate);
            rotate_image_cw(out, rotate);
            //show_image_normalized(in, "in2");
            //show_image_normalized(out, "out2");
            //cvWaitKey(0);
        }
        float loss = train_network(net, batch);
        free_data(batch);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);

            free_data(train);
            sprintf(buff, "/home/pjreddie/go.train.%02d", epoch%10);
            train = load_go(buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free(base);
    free_data(train);
}

void propagate_liberty(float *board, int *lib, int *visited, int row, int col, int num, int side)
{
    if (!num) return;
    if (row < 0 || row > 18 || col < 0 || col > 18) return;
    int index = row*19 + col;
    if (board[index] != side) return;
    if (visited[index]) return;
    visited[index] = 1;
    lib[index] += num;
    propagate_liberty(board, lib, visited, row+1, col, num, side);
    propagate_liberty(board, lib, visited, row-1, col, num, side);
    propagate_liberty(board, lib, visited, row, col+1, num, side);
    propagate_liberty(board, lib, visited, row, col-1, num, side);
}

int *calculate_liberties(float *board)
{
    int *lib = calloc(19*19, sizeof(int));
    int visited[361];
    int i, j;
    for(j = 0; j < 19; ++j){
        for(i = 0; i < 19; ++i){
            memset(visited, 0, 19*19*sizeof(int));
            int index = j*19 + i;
            if(board[index]){
                int side = board[index];
                int num = 0;
                if (i > 0  && board[j*19 + i -  1] == 0) ++num;
                if (i < 18 && board[j*19 + i +  1] == 0) ++num;
                if (j > 0  && board[j*19 + i - 19] == 0) ++num;
                if (j < 18 && board[j*19 + i + 19] == 0) ++num;
                propagate_liberty(board, lib, visited, j, i, num, side);
            }
        }
    }
    return lib;
}

void update_board(float *board)
{
    int i;
    int *l = calculate_liberties(board);
    for(i = 0; i < 19*19; ++i){
        if (board[i] < 0 && !l[i]) board[i] = 0;
    }
    free(l);
}

void print_board(float *board, int swap, int *indexes)
{
    int i,j,n;
    printf("\n\n");
    printf("   ");
    for(i = 0; i < 19; ++i){
        printf("%c ", 'A' + i + 1*(i > 7 && noi));
    }
    printf("\n");
    for(j = 0; j < 19; ++j){
        printf("%2d ", (inverted) ? 19-j : j+1);
        for(i = 0; i < 19; ++i){
            int index = j*19 + i;
            if(indexes){
                int found = 0;
                for(n = 0; n < nind; ++n){
                    if(index == indexes[n]){
                        found = 1;
                        if(n == 0) printf("\uff11");
                        else if(n == 1) printf("\uff12");
                        else if(n == 2) printf("\uff13");
                        else if(n == 3) printf("\uff14");
                        else if(n == 4) printf("\uff15");
                    }
                }
                if(found) continue;
            }
            if(board[index]*-swap > 0) printf("\u25C9 ");
            else if(board[index]*-swap < 0) printf("\u25EF ");
            else printf("  ");
        }
        printf("\n");
    }
}

void flip_board(float *board)
{
    int i;
    for(i = 0; i < 19*19; ++i){
        board[i] = -board[i];
    }
}

void test_go(char *filename, char *weightfile, int multi)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    set_batch_network(&net, 1);
    float *board = calloc(19*19, sizeof(float));
    float *move = calloc(19*19, sizeof(float));
    int color = 1;
    while(1){
        float *output = network_predict(net, board);
        copy_cpu(19*19, output, 1, move, 1);
        int i;
        if(multi){
            image bim = float_to_image(19, 19, 1, board);
            for(i = 1; i < 8; ++i){
                rotate_image_cw(bim, i);
                if(i >= 4) flip_image(bim);

                float *output = network_predict(net, board);
                image oim = float_to_image(19, 19, 1, output);

                if(i >= 4) flip_image(oim);
                rotate_image_cw(oim, -i);

                axpy_cpu(19*19, 1, output, 1, move, 1);

                if(i >= 4) flip_image(bim);
                rotate_image_cw(bim, -i);
            }
            scal_cpu(19*19, 1./8., move, 1);
        }
        for(i = 0; i < 19*19; ++i){
            if(board[i]) move[i] = 0;
        }

        int indexes[nind];
        int row, col;
        top_k(move, 19*19, nind, indexes);
        print_board(board, color, indexes);
        for(i = 0; i < nind; ++i){
            int index = indexes[i];
            row = index / 19;
            col = index % 19;
            printf("%d: %c %d, %.2f%%\n", i+1, col + 'A' + 1*(col > 7 && noi), (inverted)?19 - row : row+1, move[index]*100);
        }
        if(color == 1) printf("\u25EF Enter move: ");
        else printf("\u25C9 Enter move: ");

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
                board[row*19 + col] = 1;
            }
        } else if (cnum){
            if (c <= 'T' && c >= 'A'){
                int num = sscanf(line, "%c %d", &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 2) board[row*19 + col] = 1;
            } else if (c == 'p') {
                // Pass
            } else if(c=='b' || c == 'w'){
                char g;
                int num = sscanf(line, "%c %c %d", &g, &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 3) board[row*19 + col] = (g == 'b') ? color : -color;
            } else if(c == 'c'){
                char g;
                int num = sscanf(line, "%c %c %d", &g, &c, &row);
                row = (inverted)?19 - row : row-1;
                col = c - 'A';
                if (col > 7 && noi) col -= 1;
                if (num == 3) board[row*19 + col] = 0;
            }
        }
        free(line);
        update_board(board);
        flip_board(board);
        color = -color;
    }

}

void run_go(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    int multi = find_arg(argc, argv, "-multi");
    if(0==strcmp(argv[2], "train")) train_go(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_go(cfg, weights, multi);
}


