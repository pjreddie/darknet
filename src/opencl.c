#ifdef GPU
#include "opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


cl_info cl = {0};

void check_error(cl_info info)
{
    if (info.error != CL_SUCCESS) {
        printf("\n Error number %d", info.error);
        exit(1);
    }
}

#define MAX_DEVICES 10

cl_info cl_init()
{
    cl_info info;
    info.initialized = 0;
    cl_uint num_platforms, num_devices;
    // Fetch the Platform and Device IDs; we only want one.
    cl_device_id devices[MAX_DEVICES];
    info.error=clGetPlatformIDs(1, &info.platform, &num_platforms);
    check_error(info);
    info.error=clGetDeviceIDs(info.platform, CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
    if(num_devices > MAX_DEVICES) num_devices = MAX_DEVICES;
    int index = getpid()%num_devices;
    printf("%d rand, %d devices, %d index\n", getpid(), num_devices, index);
    //info.device = devices[index];
    info.device = devices[1];
    fprintf(stderr, "Found %d device(s)\n", num_devices);
    check_error(info);

    cl_context_properties properties[]={
	    CL_CONTEXT_PLATFORM, (cl_context_properties)info.platform,
	    0};
    // Note that nVidia's OpenCL requires the platform property
    info.context=clCreateContext(properties, 1, &info.device, 0, 0, &info.error);
    check_error(info);
    info.queue = clCreateCommandQueue(info.context, info.device, 0, &info.error);
    check_error(info);
    info.initialized = 1;
    return info;
}

cl_program cl_fprog(char *filename, char *options, cl_info info)
{
	size_t srcsize;
	char src[8192];
	memset(src, 0, 8192);
	FILE *fil=fopen(filename,"r");
	srcsize=fread(src, sizeof src, 1, fil);
	fclose(fil);
	const char *srcptr[]={src};
	// Submit the source code of the example kernel to OpenCL
	cl_program prog=clCreateProgramWithSource(info.context,1, srcptr, &srcsize, &info.error);
	check_error(info);
	char build_c[4096];
	// and compile it (after this we could extract the compiled version)
	info.error=clBuildProgram(prog, 0, 0, options, 0, 0);
	if ( info.error != CL_SUCCESS ) {
		fprintf(stderr, "Error Building Program: %d\n", info.error);
		clGetProgramBuildInfo( prog, info.device, CL_PROGRAM_BUILD_LOG, 4096, build_c, 0);
		fprintf(stderr, "Build Log for %s program:\n%s\n", filename, build_c);
	}
	check_error(info);
	return prog;
}

void cl_setup()
{
	if(!cl.initialized){
		cl = cl_init();
	}
}

cl_kernel get_kernel(char *filename, char *kernelname, char *options)
{
	cl_setup();
	cl_program prog = cl_fprog(filename, options, cl);
	cl_kernel kernel=clCreateKernel(prog, kernelname, &cl.error);
	check_error(cl);
	return kernel;
}

void cl_read_array(cl_mem mem, float *x, int n)
{
    cl_setup();
    clEnqueueReadBuffer(cl.queue, mem, CL_TRUE, 0, sizeof(float)*n,x,0,0,0);
    check_error(cl);
}

void cl_write_array(cl_mem mem, float *x, int n)
{
    cl_setup();
    clEnqueueWriteBuffer(cl.queue, mem, CL_TRUE, 0,sizeof(float)*n,x,0,0,0);
    check_error(cl);
}

void cl_copy_array(cl_mem src, cl_mem dst, int n)
{
    cl_setup();
    clEnqueueCopyBuffer(cl.queue, src, dst, 0, 0, sizeof(float)*n,0,0,0);
    check_error(cl);
}

cl_mem cl_make_array(float *x, int n)
{
    cl_setup();
    cl_mem mem = clCreateBuffer(cl.context,
            CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(float)*n, x, &cl.error);
    check_error(cl);
    return mem;
}

#endif
