#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "opencl.h"
#include "mini_blas.h"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define BLOCK 8

cl_kernel get_gemm_kernel()
{
    static int init = 0;
    static cl_kernel gemm_kernel;
    if(!init){
        gemm_kernel = get_kernel("src/gemm.cl", "gemm", "-D BLOCK=" STR(BLOCK) );
        init = 1;
    }
    return gemm_kernel;
}

void gpu_gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    cl_setup();
    cl_kernel gemm_kernel = get_gemm_kernel();
    cl_context context = cl.context;
    cl_command_queue queue = cl.queue;

    size_t size = sizeof(float)*(TA ? lda*K:lda*M);
    cl_mem A_gpu = clCreateBuffer(context,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            size, A, &cl.error);
    check_error(cl);

    size = sizeof(float)*(TB ? ldb*N:ldb*K);
    cl_mem B_gpu = clCreateBuffer(context,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            size, B, &cl.error);
    check_error(cl);

    size = sizeof(float)*(ldc*M);
    cl_mem C_gpu = clCreateBuffer(context,
            CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
            size, C, &cl.error);
    check_error(cl);

    cl_uint i = 0;
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(TA), (void*) &TA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(TB), (void*) &TB);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(M), (void*) &M);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(K), (void*) &K);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(A_gpu), (void*) &A_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(lda), (void*) &lda);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(B_gpu), (void*) &B_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ldb), (void*) &ldb);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(BETA), (void*) &BETA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(C_gpu), (void*) &C_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ldc), (void*) &ldc);
    check_error(cl);

    const size_t global_size[] = {ceil((float)M/BLOCK)*BLOCK, ceil((float)N/BLOCK)*BLOCK};
    const size_t local_size[] = {BLOCK, BLOCK};
    //printf("%zd %zd %zd %zd\n", global_size[0], global_size[1], local_size[0], local_size[1]);

    clEnqueueNDRangeKernel(queue, gemm_kernel, 2, 0, global_size, local_size, 0, 0, 0);
    check_error(cl);
    clEnqueueReadBuffer(queue, C_gpu, CL_TRUE, 0, size, C, 0, 0, 0);
    check_error(cl);
    
    clReleaseMemObject(A_gpu);
    clReleaseMemObject(B_gpu);
    clReleaseMemObject(C_gpu);

}

/*
cl_kernel get_gemm_kernel_slow()
{
    static int init = 0;
    static cl_kernel gemm_kernel;
    if(!init){
        gemm_kernel = get_kernel("src/gemm.cl", "gemm_slow");
        init = 1;
    }
    return gemm_kernel;
}

void gpu_gemm_slow(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    cl_setup();
    cl_kernel gemm_kernel = get_gemm_kernel_slow();
    cl_context context = cl.context;
    cl_command_queue queue = cl.queue;

    size_t size = sizeof(float)*(TA ? lda*K:lda*M);
    cl_mem A_gpu = clCreateBuffer(context,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            size, A, &cl.error);
    check_error(cl);

    size = sizeof(float)*(TB ? ldb*N:ldb*K);
    cl_mem B_gpu = clCreateBuffer(context,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            size, B, &cl.error);
    check_error(cl);

    size = sizeof(float)*(ldc*M);
    cl_mem C_gpu = clCreateBuffer(context,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            size, C, &cl.error);
    check_error(cl);

    cl_uint i = 0;
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(TA), (void*) &TA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(TB), (void*) &TB);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(M), (void*) &M);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(K), (void*) &K);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(A_gpu), (void*) &A_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(lda), (void*) &lda);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(B_gpu), (void*) &B_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ldb), (void*) &ldb);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(BETA), (void*) &BETA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(C_gpu), (void*) &C_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ldc), (void*) &ldc);
    check_error(cl);

    const size_t global_size[] = {M, N};

    clEnqueueNDRangeKernel(queue, gemm_kernel, 2, 0, global_size, 0, 0, 0, 0);
    clEnqueueReadBuffer(queue, C_gpu, CL_TRUE, 0, size, C, 0, 0, 0);
    
    clReleaseMemObject(A_gpu);
    clReleaseMemObject(B_gpu);
    clReleaseMemObject(C_gpu);

}
*/
