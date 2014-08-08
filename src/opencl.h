#ifdef GPU
#ifndef OPENCL_H
#define OPENCL_H
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

typedef struct {
    int initialized;
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
}cl_info;

extern cl_info cl;

void cl_setup();
void check_error(cl_info info);
cl_kernel get_kernel(char *filename, char *kernelname, char *options);
void cl_read_array(cl_mem mem, float *x, int n);
void cl_write_array(cl_mem mem, float *x, int n);
cl_mem cl_make_array(float *x, int n);
void cl_copy_array(cl_mem src, cl_mem dst, int n);
cl_mem cl_sub_array(cl_mem src, int offset, int size);
#endif
#endif
