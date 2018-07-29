/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef INCLUDED_DLAERROR_H
#define INCLUDED_DLAERROR_H

typedef enum
{
    /** File related error codes */
    NvDlaError_FileWriteFailed     = 0x00030000,
    NvDlaError_FileReadFailed      = 0x00030001,
    NvDlaError_EndOfFile           = 0x00030002,
    NvDlaError_FileOperationFailed = 0x00030003,
    NvDlaError_DirOperationFailed  = 0x00030004,
    NvDlaError_EndOfDirList        = 0x00030005,

    /** Ioctl error codes */
    NvDlaError_IoctlFailed         = 0x0003000f,
    NvDlaError_PathAlreadyExists   = 0x00030014,
    NvDlaError_SurfaceNotSupported = 0x00010003,

    /** common error codes */
    NvDlaError_Success             = 0x00000000,
    NvDlaError_NotImplemented      = 0x00000001,
    NvDlaError_NotSupported        = 0x00000002,
    NvDlaError_NotInitialized      = 0x00000003,
    NvDlaError_BadParameter        = 0x00000004,
    NvDlaError_Timeout             = 0x00000005,
    NvDlaError_InsufficientMemory  = 0x00000006,
    NvDlaError_ReadOnlyAttribute   = 0x00000007,
    NvDlaError_InvalidState        = 0x00000008,
    NvDlaError_InvalidAddress      = 0x00000009,
    NvDlaError_InvalidSize         = 0x0000000A,
    NvDlaError_BadValue            = 0x0000000B,
    NvDlaError_AlreadyAllocated    = 0x0000000D,
    NvDlaError_Busy                = 0x0000000E,
    NvDlaError_ModuleNotPresent    = 0x000a000E,
    NvDlaError_ResourceError       = 0x0000000F,
    NvDlaError_CountMismatch       = 0x00000010,
    NvDlaError_OverFlow            = 0x00000011,
    NvDlaError_Disconnected        = 0x00000012,
    NvDlaError_FileNotFound        = 0x00000013,
    NvDlaError_TestApplicationFailed  = 0x00000014,
    NvDlaError_DeviceNotFound      = 0x00000015,

    NvDlaSuccess                   = NvDlaError_Success,
    NvDlaError_Force32             = 0x7FFFFFFF

} NvDlaError;

#endif // INCLUDED_DLAERROR_H
