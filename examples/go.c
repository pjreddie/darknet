#include "darknet.h"

#include <unistd.h>

int inverted = 1;
int noi = 1;
static const int nind = 2;

typedef struct {
    char **data;
    int n;
} moves;

char *fgetgo(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 94;
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
    while((line = fgetgo(fp))){
        if(count >= m.n){
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
    //memset(board, 0, 1*19*19*sizeof(float));
    int count = 0;
    for(i = 0; i < 91; ++i){
        char c = s[i];
        for(j = 0; j < 4; ++j){
            int me = (c >> (2*j)) & 1;
            int you = (c >> (2*j + 1)) & 1;
            if (me) board[count] = 1;
            else if (you) board[count] = -1;
            else board[count] = 0;
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
            int you = (board[count] == -1);
            if (me) s[i] = s[i] | (1<<(2*j));
            if (you) s[i] = s[i] | (1<<(2*j + 1));
            ++count;
            if(count >= 19*19) break;
        }
    }
}

data random_go_moves(moves m, int n)
{
    data d = {0};
    d.X = make_matrix(n, 19*19);
    d.y = make_matrix(n, 19*19+1);
    int i;
    for(i = 0; i < n; ++i){
        float *board = d.X.vals[i];
        float *label = d.y.vals[i];
        char *b = m.data[rand()%m.n];
        int row = b[0];
        int col = b[1];
        if(row >= 19 || col >= 19){
            label[19*19] = 1;
        } else {
            label[col + 19*row] = 1;
            string_to_board(b+2, board);
            if(board[col + 19*row]) printf("hey\n");
        }

        int flip = rand()%2;
        int rotate = rand()%4;
        image in = float_to_image(19, 19, 1, board);
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
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i].learning_rate *= ngpus;
    }
    network net = nets[0];
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    char *backup_directory = "/home/pjreddie/backup/";

    char buff[256];
    moves m = load_go_moves(filename);
    //moves m = load_go_moves("games.txt");

    int N = m.n;
    printf("Moves: %d\n", N);
    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        clock_t time=clock();

        data train = random_go_moves(m, net.batch*net.subdivisions*ngpus);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        free_data(train);

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
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

void propagate_liberty(float *board, int *lib, int *visited, int row, int col, int side)
{
    if (row < 0 || row > 18 || col < 0 || col > 18) return;
    int index = row*19 + col;
    if (board[index] != side) return;
    if (visited[index]) return;
    visited[index] = 1;
    lib[index] += 1;
    propagate_liberty(board, lib, visited, row+1, col, side);
    propagate_liberty(board, lib, visited, row-1, col, side);
    propagate_liberty(board, lib, visited, row, col+1, side);
    propagate_liberty(board, lib, visited, row, col-1, side);
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
            if(board[index] == 0){
                if ((i > 0)  && board[index - 1]) propagate_liberty(board, lib, visited, j, i-1, board[index-1]);
                if ((i < 18) && board[index + 1]) propagate_liberty(board, lib, visited, j, i+1, board[index+1]);
                if ((j > 0)  && board[index - 19]) propagate_liberty(board, lib, visited, j-1, i, board[index-19]);
                if ((j < 18) && board[index + 19]) propagate_liberty(board, lib, visited, j+1, i, board[index+19]);
            }
        }
    }
    return lib;
}

void print_board(FILE *stream, float *board, int swap, int *indexes)
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
                        if(n == 0) fprintf(stream, " 1");
                        else if(n == 1) fprintf(stream, " 2");
                        else if(n == 2) fprintf(stream, " 3");
                        else if(n == 3) fprintf(stream, " 4");
                        else if(n == 4) fprintf(stream, " 5");
                    }
                }
                if(found) continue;
            }
            //if(board[index]*-swap > 0) fprintf(stream, "\u25C9 ");
            //else if(board[index]*-swap < 0) fprintf(stream, "\u25EF ");
            if(board[index]*-swap > 0) fprintf(stream, " O");
            else if(board[index]*-swap < 0) fprintf(stream, " X");
            else fprintf(stream, "  ");
        }
        fprintf(stream, "\n");
    }
}

void flip_board(float *board)
{
    int i;
    for(i = 0; i < 19*19; ++i){
        board[i] = -board[i];
    }
}

void predict_move(network net, float *board, float *move, int multi)
{
    float *output = network_predict(net, board);
    copy_cpu(19*19+1, output, 1, move, 1);
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

            axpy_cpu(19*19+1, 1, output, 1, move, 1);

            if(i >= 4) flip_image(bim);
            rotate_image_cw(bim, -i);
        }
        scal_cpu(19*19+1, 1./8., move, 1);
    }
    for(i = 0; i < 19*19; ++i){
        if(board[i]) move[i] = 0;
    }
}

void remove_connected(float *b, int *lib, int p, int r, int c)
{
    if (r < 0 || r >= 19 || c < 0 || c >= 19) return;
    if (b[r*19 + c] != p) return;
    if (lib[r*19 + c] != 1) return;
    b[r*19 + c] = 0;
    remove_connected(b, lib, p, r+1, c);
    remove_connected(b, lib, p, r-1, c);
    remove_connected(b, lib, p, r, c+1);
    remove_connected(b, lib, p, r, c-1);
}


void move_go(float *b, int p, int r, int c)
{
    int *l = calculate_liberties(b);
    b[r*19 + c] = p;
    remove_connected(b, l, -p, r+1, c);
    remove_connected(b, l, -p, r-1, c);
    remove_connected(b, l, -p, r, c+1);
    remove_connected(b, l, -p, r, c-1);
    free(l);
}

int makes_safe_go(float *b, int *lib, int p, int r, int c){
    if (r < 0 || r >= 19 || c < 0 || c >= 19) return 0;
    if (b[r*19 + c] == -p){
        if (lib[r*19 + c] > 1) return 0;
        else return 1;
    }
    if (b[r*19 + c] == 0) return 1;
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

int legal_go(float *b, char *ko, int p, int r, int c)
{
    if (b[r*19 + c]) return 0;
    char curr[91];
    char next[91];
    board_to_string(curr, b);
    move_go(b, p, r, c);
    board_to_string(next, b);
    string_to_board(curr, b);
    if(memcmp(next, ko, 91) == 0) return 0;
    return 1;
}

int generate_move(network net, int player, float *board, int multi, float thresh, float temp, char *ko, int print)
{
    int i, j;
    int empty = 1;
    for(i = 0; i < 19*19; ++i){
        if (board[i]) {
            empty = 0;
            break;
        }
    }
    if(empty) {
        return 72;
    }
    for(i = 0; i < net.n; ++i) net.layers[i].temperature = temp;

    float move[362];
    if (player < 0) flip_board(board);
    predict_move(net, board, move, multi);
    if (player < 0) flip_board(board);


    for(i = 0; i < 19; ++i){
        for(j = 0; j < 19; ++j){
            if (!legal_go(board, ko, player, i, j)) move[i*19 + j] = 0;
        }
    }

    int indexes[nind];
    top_k(move, 19*19+1, nind, indexes);
    if(thresh > move[indexes[0]]) thresh = move[indexes[nind-1]];

    for(i = 0; i < 19*19+1; ++i){
        if (move[i] < thresh) move[i] = 0;
    }


    int max = max_index(move, 19*19+1);
    int row = max / 19;
    int col = max % 19;
    int index = sample_array(move, 19*19+1);

    if(print){
        top_k(move, 19*19+1, nind, indexes);
        for(i = 0; i < nind; ++i){
            if (!move[indexes[i]]) indexes[i] = -1;
        }
        print_board(stderr, board, player, indexes);
        for(i = 0; i < nind; ++i){
            fprintf(stderr, "%d: %f\n", i+1, move[indexes[i]]);
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

void valid_go(char *cfgfile, char *weightfile, int multi, char *filename)
{
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    float *board = calloc(19*19, sizeof(float));
    float *move = calloc(19*19+1, sizeof(float));
   // moves m = load_go_moves("/home/pjreddie/backup/go.test");
    moves m = load_go_moves(filename);

    int N = m.n;
    int i;
    int correct = 0;
    for(i = 0; i <N; ++i){
        char *b = m.data[i];
        int row = b[0];
        int col = b[1];
        int truth = col + 19*row;
        string_to_board(b+2, board);
        predict_move(net, board, move, multi);
        int index = max_index(move, 19*19);
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
            if(board[j*19 + i] == 1) fprintf(fp, "play black %c%d\n", 'A'+i+(i>=8), 19-j);
            if(board[j*19 + i] == -1) fprintf(fp, "play white %c%d\n", 'A'+i+(i>=8), 19-j);
            if(board[j*19 + i]) ++count;
        }
    }
    return count;
}

void engine_go(char *filename, char *weightfile, int multi)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    set_batch_network(&net, 1);
    float *board = calloc(19*19, sizeof(float));
    char *one = calloc(91, sizeof(char));
    char *two = calloc(91, sizeof(char));
    int passed = 0;
    while(1){
        char buff[256];
        int id = 0;
        int has_id = (scanf("%d", &id) == 1);
        scanf("%s", buff);
        if (feof(stdin)) break;
        char ids[256];
        sprintf(ids, "%d", id);
        //fprintf(stderr, "%s\n", buff);
        if (!has_id) ids[0] = 0;
        if (!strcmp(buff, "protocol_version")){
            printf("=%s 2\n\n", ids);
        } else if (!strcmp(buff, "name")){
            printf("=%s DarkGo\n\n", ids);
        } else if (!strcmp(buff, "time_settings") || !strcmp(buff, "time_left")){
            char *line = fgetl(stdin);
            free(line);
            printf("=%s \n\n", ids);
        } else if (!strcmp(buff, "version")){
            printf("=%s 1.0. Want more DarkGo? You can find me on OGS, unlimited games, no waiting! https://online-go.com/user/view/434218\n\n", ids);
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
                memset(board, 0, 19*19*sizeof(float));
                printf("=%s \n\n", ids);
            }
        } else if (!strcmp(buff, "fixed_handicap")){
            int handicap = 0;
            scanf("%d", &handicap);
            int indexes[] = {72, 288, 300, 60, 180, 174, 186, 66, 294};
            int i;
            for(i = 0; i < handicap; ++i){
                board[indexes[i]] = 1;   
            }
        } else if (!strcmp(buff, "clear_board")){
            passed = 0;
            memset(board, 0, 19*19*sizeof(float));
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
                continue;
            } else {
                passed = 0;
            }
            if(c >= 'A' && c <= 'Z') c = c - 'A';
            if(c >= 'a' && c <= 'z') c = c - 'a';
            if(c >= 8) --c;
            r = 19 - r;
            fprintf(stderr, "move: %d %d\n", r, c);

            char *swap = two;
            two = one;
            one = swap;
            move_go(board, player, r, c);
            board_to_string(one, board);

            printf("=%s \n\n", ids);
            //print_board(stderr, board, 1, 0);
        } else if (!strcmp(buff, "genmove") || !strcmp(buff, "genmove_black") || !strcmp(buff, "genmove_white")){
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

            int index = generate_move(net, player, board, multi, .4, 1, two, 0);
            if(passed || index < 0){
                printf("=%s pass\n\n", ids);
                passed = 0;
            } else {
                int row = index / 19;
                int col = index % 19;

                char *swap = two;
                two = one;
                one = swap;

                move_go(board, player, row, col);
                board_to_string(one, board);
                row = 19 - row;
                if (col >= 8) ++col;
                printf("=%s %c%d\n\n", ids, 'A' + col, row);
                //print_board(board, 1, 0);
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
        } else {
            char *line = fgetl(stdin);
            free(line);
            printf("?%s unknown command\n\n", ids);
        }
        fflush(stdout);
        fflush(stderr);
    }
}

void test_go(char *cfg, char *weights, int multi)
{
    network net = parse_network_cfg(cfg);
    if(weights){
        load_weights(&net, weights);
    }
    srand(time(0));
    set_batch_network(&net, 1);
    float *board = calloc(19*19, sizeof(float));
    float *move = calloc(19*19+1, sizeof(float));
    int color = 1;
    while(1){
        int i;
        predict_move(net, board, move, multi);

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
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }

    network net2 = net;
    if(f2){
        net2 = parse_network_cfg(f2);
        if(w2){
            load_weights(&net2, w2);
        }
    }
    srand(time(0));
    char boards[600][93];
    int count = 0;
    set_batch_network(&net, 1);
    set_batch_network(&net2, 1);
    float *board = calloc(19*19, sizeof(float));
    char *one = calloc(91, sizeof(char));
    char *two = calloc(91, sizeof(char));
    int done = 0;
    int player = 1;
    int p1 = 0;
    int p2 = 0;
    int total = 0;
    while(1){
        if (done){
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
            memset(board, 0, 19*19*sizeof(float));
            player = 1;
            done = 0;
            count = 0;
            fflush(stdout);
            fflush(stderr);
        }
        print_board(stderr, board, 1, 0);
        //sleep(1);
        network use = ((total%2==0) == (player==1)) ? net : net2;
        int index = generate_move(use, player, board, multi, .4, 1, two, 0);
        if(index < 0){
            done = 1;
            continue;
        }
        int row = index / 19;
        int col = index % 19;

        char *swap = two;
        two = one;
        one = swap;

        if(player < 0) flip_board(board);
        boards[count][0] = row;
        boards[count][1] = col;
        board_to_string(boards[count] + 2, board);
        if(player < 0) flip_board(board);
        ++count;

        move_go(board, player, row, col);
        board_to_string(one, board);

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
    if(0==strcmp(argv[2], "train")) train_go(cfg, weights, c2, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) valid_go(cfg, weights, multi, c2);
    else if(0==strcmp(argv[2], "self")) self_go(cfg, weights, c2, w2, multi);
    else if(0==strcmp(argv[2], "test")) test_go(cfg, weights, multi);
    else if(0==strcmp(argv[2], "engine")) engine_go(cfg, weights, multi);
}


