#include "mini_blas.h"

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

#ifdef GPU
#include "opencl.h"

cl_kernel get_axpy_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/axpy.cl", "axpy", 0);
        init = 1;
    }
    return kernel;
}

cl_kernel get_copy_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/axpy.cl", "copy", 0);
        init = 1;
    }
    return kernel;
}

cl_kernel get_scal_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/axpy.cl", "scal", 0);
        init = 1;
    }
    return kernel;
}


void axpy_ongpu(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY)
{
    cl_setup();
    cl_kernel kernel = get_axpy_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*) &Y);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);
    check_error(cl);

    const size_t global_size[] = {N};

    clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);

}
void copy_ongpu(int N, cl_mem X, int INCX, cl_mem Y, int INCY)
{
    cl_setup();
    cl_kernel kernel = get_copy_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*) &Y);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);
    check_error(cl);

    const size_t global_size[] = {N};

    clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}
void scal_ongpu(int N, float ALPHA, cl_mem X, int INCX)
{
    cl_setup();
    cl_kernel kernel = get_scal_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
    check_error(cl);

    const size_t global_size[] = {N};

    clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}
#endif
