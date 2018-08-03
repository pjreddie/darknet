#include <stdio.h>
#include <assert.h>

#include "nvdla/IRuntime.h"
#include "nvdla/half.h"

using namespace nvdla;

extern "C" void *odla_create_runtime(void)
{
    return (void *)nvdla::createRuntime();
}

extern "C" void odla_load_loadable(void *runtime, const char *loadable, int instance)
{
    FILE *fp;
    ssize_t file_size;
    void *data;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    //read in loadable in memory
    fp = fopen(loadable, "r+");
    fseek(fp, 0L, SEEK_END);
    file_size = ftell(fp);
    data = calloc(file_size, sizeof(uint8_t));
    fseek(fp, 0L, SEEK_SET);
    fread(data, 1, file_size, fp);
    fclose(fp);

    //load loadable in runtime
    odla_runtime->load((NvU8*)data, instance);
}

extern "C" void odla_execute(void *runtime, int instance)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;

    odla_runtime->initEMU();
    odla_runtime->submit();
    odla_runtime->stopEMU();
}

extern "C" int odla_num_input(void *runtime)
{
    int num_input;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    odla_runtime->getNumInputTensors(&num_input);

    return num_input;
}

extern "C" int odla_num_output(void *runtime)
{
    int num_output;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    odla_runtime->getNumOutputTensors(&num_output);

    return num_output;
}

extern "C" void odla_alloc_input_tensor(void *runtime, void **buffer, int index)
{
    void *hMem = NULL;
    int err = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    err = odla_runtime->getInputTensorDesc(index, &tDesc);
    if (err)
        fprintf(stderr, "getInputTensorDesc failed\n");
    err = odla_runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, buffer);
    if (err)
        fprintf(stderr, "allocateSystemMemory failed\n");
    err = odla_runtime->bindInputTensor(index, hMem);
    if (!err)
        fprintf(stderr, "bindInputTensor failed\n");
}

extern "C" void odla_alloc_output_tensor(void *runtime, void **buffer, int index)
{
    void *hMem = NULL;
    int err = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    err = odla_runtime->getOutputTensorDesc(index, &tDesc);
    if (err)
        fprintf(stderr, "getOutputTensorDesc failed\n");
    err = odla_runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, buffer);
    if (err)
        fprintf(stderr, "allocateSystemMemory failed\n");
    err = odla_runtime->bindOutputTensor(index, hMem);
    if (err)
        fprintf(stderr, "bindOutputTensor failed\n");
}

extern "C" int odla_input_channel(void *runtime, int index)
{
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    odla_runtime->getInputTensorDesc(index, &tDesc);
    return tDesc.dims.c;
}

extern "C" int odla_input_width(void *runtime, int index)
{
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    odla_runtime->getInputTensorDesc(index, &tDesc);
    return tDesc.dims.w;
}

extern "C" int odla_input_height(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    odla_runtime->getInputTensorDesc(index, &tDesc);
    return tDesc.dims.h;
}

extern "C" int odla_output_channel(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    odla_runtime->getOutputTensorDesc(index, &tDesc);
    return tDesc.dims.c;
}

extern "C" int odla_output_width(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    odla_runtime->getOutputTensorDesc(index, &tDesc);
    return tDesc.dims.w;
}

extern "C" int odla_output_height(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    odla_runtime->getOutputTensorDesc(index, &tDesc);
    return tDesc.dims.h;
}

extern "C" void odla_copy_input(float *input, uint32_t size, void *buffer)
{
    unsigned int i = 0;
    uint8_t *out_buf = static_cast<uint8_t*>(buffer);
    unsigned int offset = 0;

    for (i = 0; i < size; i++) {
        half_float::half* outp = reinterpret_cast<half_float::half*>(out_buf+offset);
        offset += 2;
        *outp = half_float::half(*input);
        input++;
    }
}

extern "C" void odla_copy_output(void *buffer, uint32_t size, float *output)
{
    unsigned int j = 0;
    half_float::half* outp = reinterpret_cast<half_float::half*>(buffer);

    for (j = 0; j < size; j++) {
        *output = (float)*outp;
        output++;
        outp++;
    }
}
