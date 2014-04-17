

__kernel void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int lda, 
                    __global float *B, int ldb,
                    float BETA,
                    __global float *C, int ldc)
{
    __local float Asub[BLOCK][BLOCK];
    __local float Bsub[BLOCK][BLOCK];

    float val = 0;
    
    int row_block = get_group_id(0);
    int col_block = get_group_id(1);

    int sub_row = get_local_id(0);
    int sub_col = get_local_id(1);

    int row = row_block*BLOCK + sub_row;
    int col = col_block*BLOCK + sub_col;

    int i,j;
    for(i = 0; i < K; i += BLOCK){
        int arow = row_block*BLOCK + sub_row;
        int acol = i + sub_col;

        int brow = i + sub_row;
        int bcol = col_block*BLOCK + sub_col;

        Asub[sub_row][sub_col] = TA ? A[arow + acol*lda] : A[arow*lda + acol];
        Bsub[sub_row][sub_col] = TB ? B[brow + bcol*ldb] : B[brow*ldb + bcol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < BLOCK && i+j<K; ++j){
            val += Asub[sub_row][j]*Bsub[j][sub_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row < M && col < N){
        C[row*ldc+col] = val;
    }
}

/*
__kernel void gemm_slow(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int lda, 
                    __global float *B, int ldb,
                    float BETA,
                    __global float *C, int ldc)
{
    float val = 0;
    int row = get_global_id(0);
    int col = get_global_id(1);
    int i;
    for(i = 0; i < K; ++i){
        float Aval;
        if(TA) Aval = A[i*lda+row]; 
        else Aval = A[row*lda+i];

        float Bval;
        if(TB) Bval = B[col*ldb+i];
        else Bval = B[col+i*ldb];

        val += Aval*Bval;
    }
    C[row*ldc+col] = val;
}

*/
