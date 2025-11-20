// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/library_types.h>

namespace mori {
namespace moe {

inline const char* HipDataTypeToString(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_16F:
      return "HIP_R_16F";
    case HIP_R_32F:
      return "HIP_R_32F";
    case HIP_R_16BF:
      return "HIP_R_16BF";
    case HIP_R_8F_E4M3_FNUZ:
      return "HIP_R_8F_E4M3_FNUZ";
    default:
      return "Unknown";
  }
}

inline size_t GetHipDataTypeSize(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_32F:
      return sizeof(float);
    case HIP_R_16BF:
      return sizeof(hip_bfloat16);
    case HIP_R_8F_E4M3_FNUZ:
      return sizeof(__hip_fp8_e4m3_fnuz);
    default:
      throw std::runtime_error("Unknown hipDataType");
  }
}

using index_t = int32_t;

}  // namespace moe
}  // namespace mori
