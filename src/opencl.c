#include "opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cl_info cl = {0};

void check_error(cl_info info)
{
    if (info.error != CL_SUCCESS) {
        printf("\n Error number %d", info.error);
    }
}

cl_info cl_init()
{
    cl_info info;
    info.initialized = 0;
    cl_uint platforms, devices;
    // Fetch the Platform and Device IDs; we only want one.
    info.error=clGetPlatformIDs(1, &info.platform, &platforms);
    check_error(info);
    info.error=clGetDeviceIDs(info.platform, CL_DEVICE_TYPE_ALL, 1, &info.device, &devices);
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


