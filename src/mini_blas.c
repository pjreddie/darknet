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

//This one might be too, can't remember.
void col2im_cpu(float* data_col, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_im) 
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for ( c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for ( h = 0; h < height_col; ++h) {
            for ( w = 0; w < width_col; ++w) {
                data_im[(c_im * height + h * stride + h_offset) * width
                    + w * stride + w_offset]+= data_col[(c * height_col + h) * width_col + w];
            }
        }
    }
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
    for(i = 0; i<1000; ++i){
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

