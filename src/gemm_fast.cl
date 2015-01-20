
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

    float Areg[TILE];
    float acc[TILE][TILE/THREADS];

    A += rtile*TILE*lda;
    B += ctile*TILE;
    C += rtile*TILE*ldc + ctile*TILE;

    for(i = 0; i < TILE; ++i){
        for(j = 0; j < TILE/THREADS; ++j){
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
            #pragma unroll
            for(y = 0; y < TILE; ++y){
                Areg[y] = Asub[y][k];
            }
            for(x = 0; x < TILE; x += THREADS){
                float Breg = Bsub[k][x+offset];
                #pragma unroll
                for(y = 0; y < TILE; ++y){
                    acc[y][x/THREADS] += Breg * Areg[y];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(i = 0; i < TILE; ++i){
        for(j = 0; j < TILE/THREADS; ++j){
            int col = j*THREADS + offset;
            int row = i;
            C[row*ldc+col] = ALPHA*acc[i][j] + BETA*C[row*ldc+col];
        }
    }
}

