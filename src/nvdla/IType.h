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

#ifndef NVDLA_I_TYPE_H
#define NVDLA_I_TYPE_H

#include "dlaerror.h"
#include "dlatypes.h"

#include "c/NvDlaType.h"

namespace nvdla
{

class Dims4
{
public:
    Dims4() : n(1), c(0), h(0), w(0) {};
    Dims4(NvS32 c, NvS32 h, NvS32 w) : n(1), c(c), h(h), w(w) {};
    Dims4(NvS32 n, NvS32 c, NvS32 h, NvS32 w) : n(n), c(c), h(h), w(w) {};
    NvS32 n;      //!< the number of images in the data or number of kernels in the weights (default = 1)
    NvS32 c;      //!< the number of channels in the data
    NvS32 h;      //!< the height of the data
    NvS32 w;      //!< the width of the data
    inline bool operator==(const Dims4& other) const
    {
        return (n == other.n && c == other.c && h == other.h && w == other.w);
    }
    inline bool operator!=(const Dims4& other) const
    {
        return !(n == other.n && c == other.c && h == other.h && w == other.w);
    }
};


} // nvdla::

#endif // NVDLA_I_TYPE_H
