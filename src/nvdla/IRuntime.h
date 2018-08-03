/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVDLA_I_RUNTIME_H
#define NVDLA_I_RUNTIME_H

#include "dlaerror.h"
#include "dlatypes.h"

#include "NvDlaType.h"

namespace nvdla
{

class IRuntime
{
public:

#define TENSOR_DATA_FORMAT_UNKNOWN   0U
#define TENSOR_DATA_FORMAT_NCHW      1U
#define TENSOR_DATA_FORMAT_NHWC      2U

#define TENSOR_DATA_TYPE_UNKNOWN 0U
#define TENSOR_DATA_TYPE_FLOAT   1U
#define TENSOR_DATA_TYPE_HALF    2U
#define TENSOR_DATA_TYPE_INT16   3U
#define TENSOR_DATA_TYPE_INT8    4U

#define TENSOR_PIXEL_FORMAT_R8 0U
#define TENSOR_PIXEL_FORMAT_R10 1U
#define TENSOR_PIXEL_FORMAT_R12 2U
#define TENSOR_PIXEL_FORMAT_R16 3U
#define TENSOR_PIXEL_FORMAT_R16_I 4U
#define TENSOR_PIXEL_FORMAT_R16_F 5U
#define TENSOR_PIXEL_FORMAT_A16B16G16R16 6U
#define TENSOR_PIXEL_FORMAT_X16B16G16R16 7U
#define TENSOR_PIXEL_FORMAT_A16B16G16R16_F 8U
#define TENSOR_PIXEL_FORMAT_A16Y16U16V16 9U
#define TENSOR_PIXEL_FORMAT_V16U16Y16A16 10U
#define TENSOR_PIXEL_FORMAT_A16Y16U16V16_F 11U
#define TENSOR_PIXEL_FORMAT_A8B8G8R8 12U
#define TENSOR_PIXEL_FORMAT_A8R8G8B8 13U
#define TENSOR_PIXEL_FORMAT_B8G8R8A8 14U
#define TENSOR_PIXEL_FORMAT_R8G8B8A8 15U
#define TENSOR_PIXEL_FORMAT_X8B8G8R8 16U
#define TENSOR_PIXEL_FORMAT_X8R8G8B8 17U
#define TENSOR_PIXEL_FORMAT_B8G8R8X8 18U
#define TENSOR_PIXEL_FORMAT_R8G8B8X8 19U
#define TENSOR_PIXEL_FORMAT_A2B10G10R10 20U
#define TENSOR_PIXEL_FORMAT_A2R10G10B10 21U
#define TENSOR_PIXEL_FORMAT_B10G10R10A2 22U
#define TENSOR_PIXEL_FORMAT_R10G10B10A2 23U
#define TENSOR_PIXEL_FORMAT_A2Y10U10V10 24U
#define TENSOR_PIXEL_FORMAT_V10U10Y10A2 25U
#define TENSOR_PIXEL_FORMAT_A8Y8U8V8 26U
#define TENSOR_PIXEL_FORMAT_V8U8Y8A8 27U
#define TENSOR_PIXEL_FORMAT_Y8___U8V8_N444 28U
#define TENSOR_PIXEL_FORMAT_Y8___V8U8_N444 29U
#define TENSOR_PIXEL_FORMAT_Y10___U10V10_N444 30U
#define TENSOR_PIXEL_FORMAT_Y10___V10U10_N444 31U
#define TENSOR_PIXEL_FORMAT_Y12___U12V12_N444 32U
#define TENSOR_PIXEL_FORMAT_Y12___V12U12_N444 33U
#define TENSOR_PIXEL_FORMAT_Y16___U16V16_N444 34U
#define TENSOR_PIXEL_FORMAT_Y16___V16U16_N444 35U
#define TENSOR_PIXEL_FORMAT_FEATURE 36U
#define TENSOR_PIXEL_FORMAT_UNKNOWN 37U

#define TENSOR_PIXEL_MAPPING_PITCH_LINEAR 0U


#define NVDLA_RUNTIME_TENSOR_DESC_NAME_MAX_LEN 80U  /* name string length */
#define NVDLA_RUNTIME_TENSOR_DESC_NUM_STRIDES 8U    /* a little room to grow */

/* strides are in units of bytes */
#define NVDLA_RUNTIME_TENSOR_DESC_NCHW_E_STRIDE   0U /* elem stride/bpe */
#define NVDLA_RUNTIME_TENSOR_DESC_NCHW_W_STRIDE   1U /* line            */
#define NVDLA_RUNTIME_TENSOR_DESC_NCHW_HW_STRIDE  2U /* surface         */
#define NVDLA_RUNTIME_TENSOR_DESC_NCHW_CHW_STRIDE 3U /* tensor          */

#define NVDLA_RUNTIME_TENSOR_DESC_NCxHWx_E_STRIDE     0U /* elem stride/bpe  */
#define NVDLA_RUNTIME_TENSOR_DESC_NCxHWx_X_STRIDE     1U
#define NVDLA_RUNTIME_TENSOR_DESC_NCxHWx_WX_STRIDE    2U /* line (hw pov)    */
#define NVDLA_RUNTIME_TENSOR_DESC_NCxHWx_HWX_STRIDE   3U /* surface (hw pov) */
#define NVDLA_RUNTIME_TENSOR_DESC_NCxHWx_CXHWX_STRIDE 4U /* tensor           */

#define NVDLA_RUNTIME_TENSOR_DESC_NHWC_E_STRIDE   0U /* elem stride/bpe */
#define NVDLA_RUNTIME_TENSOR_DESC_NHWC_C_STRIDE   1U /* channel   */
#define NVDLA_RUNTIME_TENSOR_DESC_NHWC_WC_STRIDE  2U /* line      */
#define NVDLA_RUNTIME_TENSOR_DESC_NHWC_HWC_STRIDE 3U /* tensor    */

    struct NvDlaTensor
    {
        char name[NVDLA_RUNTIME_TENSOR_DESC_NAME_MAX_LEN + 1];
        NvU64 bufferSize;
        NvDlaDims4 dims;
        NvU8 dataFormat;    /* _DATA_FORMAT   */
        NvU8 dataType;      /* _DATA_TYPE     */
        NvU8 dataCategory;  /* _DATA_CATEGORY */
        NvU8 pixelFormat;   /* _PIXEL_FORMAT  */
        NvU8 pixelMapping;  /* _PIXEL_MAPPING */

        NvU32 stride[NVDLA_RUNTIME_TENSOR_DESC_NUM_STRIDES];

    };
    typedef struct NvDlaTensor NvDlaTensor;

    virtual NvU16 getMaxDevices() = 0;
    virtual NvU16 getNumDevices() = 0;
    virtual bool initEMU(void) = 0;
    virtual void stopEMU(void) = 0;

    virtual bool load(NvU8 *buf, int instance) = 0;
    virtual void unload(void) = 0;
    virtual NvDlaError allocateSystemMemory(void **h_mem, NvU64 size, void **pData) = 0;
    virtual void freeSystemMemory(void *phMem, NvU64 size) = 0;

    virtual bool bindInputTensor(int index, void *hMem) = 0;
    virtual bool bindOutputTensor(int index, void *hMem) = 0;

    virtual NvDlaError getNetworkDataType(uint8_t *) const = 0;

    virtual NvDlaError getNumInputTensors(int *) = 0;
    virtual NvDlaError getInputTensorDesc(int id, NvDlaTensor *) = 0;
    virtual NvDlaError setInputTensorDesc(int id, const NvDlaTensor *) = 0;

    virtual NvDlaError getNumOutputTensors(int *) = 0;
    virtual NvDlaError getOutputTensorDesc(int id, NvDlaTensor *) = 0;
    virtual NvDlaError setOutputTensorDesc(int id, const NvDlaTensor *) = 0;

    virtual bool submit() = 0;

protected:
    IRuntime();
    virtual ~IRuntime();
};

IRuntime *createRuntime();
void destroyRuntime(IRuntime *runtime);

} // nvdla

#endif // NVDLA_I_RUNTIME_H
