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

#include "mori/ops/dispatch_combine/utils.hpp"

#include <hip/hip_runtime.h>
#include <numa.h>

#include <fstream>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace moe {

/* ---------------------------------------------------------------------------------------------- */
/*                                          GPU utils                                             */
/* ---------------------------------------------------------------------------------------------- */
void gpuFreeHost(void* ptr) { HIP_RUNTIME_CHECK(hipFree(ptr)); }

void* gpuCallocHost(size_t bytes, unsigned int flags) {
  void* ptr;
  HIP_RUNTIME_CHECK(hipHostAlloc(&ptr, bytes, flags));
  ::memset(ptr, 0, bytes);
  return ptr;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          NUMA utils                                            */
/* ---------------------------------------------------------------------------------------------- */
static const std::string getBusId(int deviceId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdChar[] = "00000000:00:00.0";
  MSCCLPP_CUDATHROW(hipDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  // we need the hex in lower case format
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

int getDeviceNumaNode(int deviceId) {
  std::string busId = getBusId(deviceId);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/numa_node";
  std::ifstream file(file_str);
  int numaNode;
  if (file.is_open()) {
    if (!(file >> numaNode)) {
      throw std::runtime_error("Failed to read NUMA node from file: " + file_str);
    }
  } else {
    throw std::runtime_error("Failed to open file: " + file_str);
  }
  return numaNode;
}

void numaBind(int node) {
  int totalNumNumaNodes = numa_num_configured_nodes();
  if (node < 0 || node >= totalNumNumaNodes) {
    throw std::runtime_error("Invalid NUMA node " + std::to_string(node) +
                             ", must be between 0 and " + std::to_string(totalNumNumaNodes));
  }
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Stream Pool                                           */
/* ---------------------------------------------------------------------------------------------- */
StreamPool::StreamPool(int npes) : streams_(npes) {
  for (int i = 0; i < npes; ++i) {
    hipStreamCreateWithFlag(&streams_[i], hipStreamNonBlocking);
  }
}

StreamPool::~StreamPool() {
  for (auto stream_ : streams_) {
    hipStreamDestroy(stream_);
  }
}

hipStream_t StreamPool::GetStream(int destPe) { return streams_[destPe]; }

}  // namespace moe
}  // namespace mori
