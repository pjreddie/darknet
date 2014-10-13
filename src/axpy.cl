__kernel void axpy(int N, float ALPHA, __global float *X, int INCX, __global float *Y, int INCY)
{
    int i = get_global_id(0);
    Y[i*INCY] += ALPHA*X[i*INCX];
}

__kernel void scal(int N, float ALPHA, __global float *X, int INCX)
{
    int i = get_global_id(0);
    X[i*INCX] *= ALPHA;
}

__kernel void copy(int N, __global float *X, int INCX, __global float *Y, int INCY)
{
    int i = get_global_id(0);
    Y[i*INCY] = X[i*INCX];
}

