#include <stdio.h>
#include <assert.h>

#include "nvdla/IRuntime.h"
#include "nvdla/half.h"

using namespace nvdla;

extern "C" void *odla_create_runtime(void)
{
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    std::cout << __func__ << __LINE__ << std::endl;
    return (void *)nvdla::createRuntime();
}

extern "C" void odla_load_loadable(void *runtime, const char *loadable, int instance)
{
    FILE *fp;
    ssize_t file_size;
    void *data;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    //read in loadable in memory
    fp = fopen(loadable, "r+");
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    fseek(fp, 0L, SEEK_END);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    file_size = ftell(fp);
    fprintf(stderr, "%s %d file size %u\n", __func__, __LINE__, file_size);
    data = calloc(file_size, sizeof(uint8_t));
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    fseek(fp, 0L, SEEK_SET);
    fread(data, 1, file_size, fp);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    fclose(fp);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);

    fprintf(stderr, "@@@ %s %d instance %d\n", __func__, __LINE__, instance);
    //load loadable in runtime
    odla_runtime->load((NvU8*)data, instance);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}

extern "C" void odla_execute(void *runtime, int instance)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->initEMU();
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->submit();
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->stopEMU();
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}

extern "C" int odla_num_input(void *runtime)
{
    int num_input;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getNumInputTensors(&num_input);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);

    return num_input;
}

extern "C" int odla_num_output(void *runtime)
{
    int num_output;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getNumOutputTensors(&num_output);

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    return num_output;
}

extern "C" void odla_alloc_input_tensor(void *runtime, void *buffer, int index)
{
    void *hMem = NULL;
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getInputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, &buffer);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->bindInputTensor(index, hMem);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}

extern "C" void odla_alloc_output_tensor(void *runtime, void *buffer, int index)
{
    void *hMem = NULL;
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getOutputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, &buffer);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->bindInputTensor(index, hMem);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}

extern "C" int odla_input_channel(void *runtime, int index)
{
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getInputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    return tDesc.dims.c;
}

extern "C" int odla_input_width(void *runtime, int index)
{
    nvdla::IRuntime::NvDlaTensor tDesc;
    IRuntime *odla_runtime = (IRuntime *)runtime;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getInputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    return tDesc.dims.w;
}

extern "C" int odla_input_height(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getInputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    return tDesc.dims.h;
}

extern "C" int odla_output_channel(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getOutputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    return tDesc.dims.c;
}

extern "C" int odla_output_width(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getOutputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    return tDesc.dims.w;
}

extern "C" int odla_output_height(void *runtime, int index)
{
    IRuntime *odla_runtime = (IRuntime *)runtime;
    nvdla::IRuntime::NvDlaTensor tDesc;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    odla_runtime->getOutputTensorDesc(index, &tDesc);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    return tDesc.dims.h;
}

extern "C" void odla_copy_input(float *input, uint32_t size, void *buffer)
{
    unsigned int i = 0;

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    for (i = 0; i < size; i++) {
        half_float::half* outp = reinterpret_cast<half_float::half*>(buffer);

        *outp = half_float::half(*input);
        outp++;
        input++;
    }
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}

extern "C" void odla_copy_output(void *buffer, uint32_t size, float *output)
{
    unsigned int j = 0;
    half_float::half* outp = reinterpret_cast<half_float::half*>(buffer);

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    for (j = 0; j < size; j++) {
        *output = (float)*outp;
        output++;
        outp++;
    }
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}
