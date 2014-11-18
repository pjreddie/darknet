#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "mini_blas.h"

void pm(int M, int N, float *A)
{
    int i,j;
    for(i =0 ; i < M; ++i){
        for(j = 0; j < N; ++j){
            printf("%10.6f, ", A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void test_blas()
{

    time_random_matrix(0,0,100,100,100); 
    time_random_matrix(1,0,100,100,100); 
    time_random_matrix(0,1,100,100,100); 
    time_random_matrix(1,1,100,100,100); 

    time_random_matrix(0,0,1000,100,100); 
    time_random_matrix(1,0,1000,100,100); 
    time_random_matrix(0,1,1000,100,100); 
    time_random_matrix(1,1,1000,100,100); 
}

