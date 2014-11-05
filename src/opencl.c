#ifdef GPU
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef CLBLAS
#include <clBLAS.h>
#endif

#include "opencl.h"
#include "utils.h"

cl_info cl = {0};

void check_error(cl_info info)
{
    clFinish(cl.queue);
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

    printf("=== %d OpenCL platform(s) found: ===\n", num_platforms);
    char buffer[10240];
    clGetPlatformInfo(info.platform, CL_PLATFORM_PROFILE, 10240, buffer, NULL);
    printf("  PROFILE = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VERSION, 10240, buffer, NULL);
    printf("  VERSION = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_NAME, 10240, buffer, NULL);
    printf("  NAME = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VENDOR, 10240, buffer, NULL);
    printf("  VENDOR = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
    printf("  EXTENSIONS = %s\n", buffer);

    check_error(info);
    info.error=clGetDeviceIDs(info.platform, CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
    if(num_devices > MAX_DEVICES) num_devices = MAX_DEVICES;
    printf("=== %d OpenCL device(s) found on platform:\n", num_devices);
    int i;
    for (i=0; i<num_devices; i++)
    {
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        printf("  -- %d --\n", i);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_NAME = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VENDOR = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DRIVER_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_MAX_WORK_GROUP_SIZE = %llu\n", (unsigned long long)buf_ulong);
        cl_uint items;
        clGetDeviceInfo( devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), 
                                       &items, NULL);
        printf("  DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", (unsigned int)items);
        size_t workitem_size[10];
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 10*sizeof(workitem_size), workitem_size, NULL);
        printf("  DEVICE_MAX_WORK_ITEM_SIZES = %u / %u / %u \n", (unsigned int)workitem_size[0], (unsigned int)workitem_size[1], (unsigned int)workitem_size[2]);

    }
    int index = getpid()%num_devices;
    index = 0;
    printf("%d rand, %d devices, %d index\n", getpid(), num_devices, index);
    info.device = devices[index];
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
    #ifdef CLBLAS
    info.error = clblasSetup();
    #endif
    check_error(info);
    info.initialized = 1;
    return info;
}

cl_program cl_fprog(char *filename, char *options, cl_info info)
{
	size_t srcsize;
	char src[64*1024];
	memset(src, 0, 64*1024);
	FILE *fil=fopen(filename,"r");
    if(fil == 0) file_error(filename);
	srcsize=fread(src, sizeof src, 1, fil);
	fclose(fil);
	const char *srcptr[]={src};
	// Submit the source code of the example kernel to OpenCL
	cl_program prog=clCreateProgramWithSource(info.context,1, srcptr, &srcsize, &info.error);
	check_error(info);
	char build_c[1024*64];
	// and compile it (after this we could extract the compiled version)
	info.error=clBuildProgram(prog, 0, 0, options, 0, 0);
	if ( info.error != CL_SUCCESS ) {
		fprintf(stderr, "Error Building Program: %d\n", info.error);
		clGetProgramBuildInfo( prog, info.device, CL_PROGRAM_BUILD_LOG, 1024*64, build_c, 0);
		fprintf(stderr, "Build Log for %s program:\n%s\n", filename, build_c);
	}
	check_error(info);
	return prog;
}

void cl_setup()
{
	if(!cl.initialized){
        printf("initializing\n");
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

cl_mem cl_sub_array(cl_mem src, int offset, int size)
{
    cl_buffer_region r;
    r.origin = offset*sizeof(float);
    r.size = size*sizeof(float);
    cl_mem sub = clCreateSubBuffer(src, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &cl.error);
    check_error(cl);
    return sub;
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

cl_mem cl_make_int_array(int *x, int n)
{
    cl_setup();
    cl_mem mem = clCreateBuffer(cl.context,
            CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(int)*n, x, &cl.error);
    check_error(cl);
    return mem;
}

#endif
