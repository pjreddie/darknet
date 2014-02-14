
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

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

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc)
{
    // Assume beta = 1 LULZ
    int i,j,k;
    if(TB && !TA){
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                register float sum = 0;
                for(k = 0; k < K; ++k){
                    sum += ALPHA*A[i*lda+k]*B[k+j*ldb];
                }
                C[i*ldc+j] += sum;
            }
        }
    }else if(TA && !TB){
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                register float A_PART = ALPHA*A[k*lda+i];
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += A_PART*B[k*ldb+j];
                }
            }
        }
    }else{
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                register float A_PART = ALPHA*A[i*lda+k];
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += A_PART*B[k*ldb+j];
                }
            }
        }
    }
}

void im2row(float *image, int h, int w, int c, int size, int stride, float *matrix)
{
    int i;
    int mc = c;
    int mw = (size*size);
    int mh = ((h-size)/stride+1)*((w-size)/stride+1);
    int msize = mc*mw*mh;
    for(i = 0; i < msize; ++i){
        int channel = i/(mh*mw);
        int block =   (i%(mh*mw))/mw;
        int position = i%mw;
        int block_h = block/((w-size)/stride+1);
        int block_w = block%((w-size)/stride+1);
        int ph, pw, pc;
        ph = position/size+block_h;
        pw = position%size+block_w;
        pc = channel;
        matrix[i] = image[pc*h*w+ph*w+pw];
    }
}
void im2col(float *image, int h, int w, int c, int size, int stride, float *matrix)
{
    int b,p;
    int blocks = ((h-size)/stride+1)*((w-size)/stride+1);
    int pixels = (size*size*c);
    for(b = 0; b < blocks; ++b){
        int block_h = b/((w-size)/stride+1);
        int block_w = b%((w-size)/stride+1);
        for(p = 0; p < pixels; ++p){
            int ph, pw, pc;
            int position = p%(size*size);
            pc = p/(size*size);
            ph = position/size+block_h;
            pw = position%size+block_w;
            matrix[b+p*blocks] = image[pc*h*w+ph*w+pw];
        }
    }
}

//From Berkeley Vision's Caffe!
void im2col_cpu(float* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_col) 
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
                data_col[(c * height_col + h) * width_col + w] =
                    data_im[(c_im * height + h * stride + h_offset) * width
                    + w * stride + w_offset];
            }
        }
    }
}

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
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);
    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<1000; ++i){
        gemm(TA,TB,m,n,k,1,a,k,b,n,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
}

void test_blas()
{
    time_random_matrix(0,0,100,100,100); 
    time_random_matrix(1,0,100,100,100); 
    time_random_matrix(0,1,100,100,100); 

    time_random_matrix(0,1,1000,100,100); 
    time_random_matrix(1,0,1000,100,100); 

}

