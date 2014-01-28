
#include <stdlib.h>
#include <math.h>

void pm(int M, int N, double *A)
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

void gemm(int TA, int TB, int M, int N, int K, double ALPHA, 
                    double *A, int lda, 
                    double *B, int ldb,
                    double BETA,
                    double *C, int ldc)
{
    // Assume TA = 0, beta = 1 LULZ
    int i,j,k;
    if(TB && !TA){
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                register double sum = 0;
                for(k = 0; k < K; ++k){
                    sum += ALPHA*A[i*lda+k]*B[k+j*ldb];
                }
                C[i*ldc+j] += sum;
            }
        }
    }else{
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                register double A_PART = ALPHA*A[i*lda+k];
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += A_PART*B[k*ldb+j];
                }
            }
        }
    }
}

void im2row(double *image, int h, int w, int c, int size, int stride, double *matrix)
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
void im2col(double *image, int h, int w, int c, int size, int stride, double *matrix)
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
void im2col_cpu(double* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride,
        double* data_col) 
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

