
__kernel void gemm_nn_fast(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int a_off, int lda, 
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc)
{
    int i, j, k, x, y;
    A += a_off;
    B += b_off;
    C += c_off;

    __local float Asub[TILE]  [TILE_K];
    __local float Bsub[TILE_K][TILE];

    int ctile = get_group_id(0);
    int rtile = get_group_id(1);

    float Breg;
    float Areg[WPT];
    float acc[WPT][WPT];

    A += rtile*TILE*lda;
    B += ctile*TILE;
    C += rtile*TILE*ldc + ctile*TILE;

    for(i = 0; i < WPT; ++i){
        for(j = 0; j < WPT; ++j){
            acc[i][j] = 0;
        }
    }

    int offset = get_local_id(0);

    for(i = 0; i < K; i += TILE_K){
        for(j = 0; j < TILE*TILE_K; j += THREADS){
            int index = j+offset;

            int row = index / TILE_K;
            int col = index % TILE_K;
            Asub[row][col] = A[row*lda + col];

            row = index / TILE;
            col = index % TILE;
            Bsub[row][col] = B[row*ldb + col];
        }

        A += TILE_K;
        B += TILE_K*ldb;

        barrier(CLK_LOCAL_MEM_FENCE);

        for(k = 0; k < TILE_K; ++k){
            for(y = 0; y < WPT; ++y){
                int row = (offset + (y*WPT)*THREADS)/TILE;
                //Areg[y] = Asub[y*WPT][k];
            }
            for(y = 0; y < WPT; ++y){
                for(x = 0; x < WPT; ++x){
                    int index = offset + (y*WPT + x)*THREADS;
                    int row = index / TILE;
                    int col = index % TILE;
                    acc[y][x] += Asub[row][k]*Bsub[k][col];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(y = 0; y < WPT; ++y){
        for(x = 0; x < WPT; ++x){
            int index = offset + (y*WPT + x)*THREADS;
            int row = index / TILE;
            int col = index % TILE;
            C[row*ldc+col] = ALPHA*acc[y][x] + BETA*C[row*ldc+col];
        }
    }
}

