__kernel void gemm_tn(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int lda, 
                    __global float *B, int ldb,
                    float BETA,
                    __global float *C, int ldc)
{
    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];

    int col = get_global_id(0);
    int row = get_global_id(1);

    int col_block = get_group_id(0);
    int row_block = get_group_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float val = 0;
    float orig = C[row*ldc + col];

    for(i = 0; i < K; i += BLOCK){
        
        int arow = y + i;
        int acol = x + row_block*BLOCK;

        int brow = y + i;
        int bcol = col;

        arow = (arow < K) ? arow : K-1;
        acol = (acol < M) ? acol : M-1;
        brow = (brow < K) ? brow : K-1;
        
        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;
        
        Asub[x][y] = A[aind];
        Bsub[y][x] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < BLOCK && i+j<K; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm_nt(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int lda, 
                    __global float *B, int ldb,
                    float BETA,
                    __global float *C, int ldc)
{
    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];

    
    int col = get_global_id(0);
    int row = get_global_id(1);

    int col_block = get_group_id(0);
    int row_block = get_group_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float val = 0;
    float orig = C[row*ldc + col];

    for(i = 0; i < K; i += BLOCK){
        
        int arow = row;
        int acol = x + i;

        int brow = col_block*BLOCK + y;
        int bcol = x + i;

        brow = (brow < N) ? brow : N-1;
        acol = (acol < K) ? acol : K-1;
        bcol = (bcol < K) ? bcol : K-1;
        
        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;
        
        Asub[y][x] = A[aind];
        Bsub[x][y] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < BLOCK && i+j<K; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int lda, 
                    __global float *B, int ldb,
                    float BETA,
                    __global float *C, int ldc)
{
    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];

    int col = get_global_id(0);
    int row = get_global_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float orig = C[row*ldc+col];
    float val = 0;
    
    for(i = 0; i < K; i += BLOCK){
        
        int arow = row;
        int acol = x + i;

        int brow = y + i;
        int bcol = col;

        acol = (acol < K) ? acol : K-1;
        brow = (brow < K) ? brow : K-1;
        
        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;
        
        Asub[y][x] = A[aind];
        Bsub[y][x] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < BLOCK && i+j<K; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

