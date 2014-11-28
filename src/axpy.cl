__kernel void axpy(int N, float ALPHA, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)
{
    int i = get_global_id(0);
    Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

__kernel void scal(int N, float ALPHA, __global float *X, int INCX)
{
    int i = get_global_id(0);
    X[i*INCX] *= ALPHA;
}

__kernel void mask(int n, __global float *x, __global float *mask, int mod)
{
    int i = get_global_id(0);
    x[i] = (mask[(i/mod)*mod]) ? x[i] : 0;
}

__kernel void copy(int N, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)
{
    int i = get_global_id(0);
    Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

