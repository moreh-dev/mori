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

#include <hip/hip_runtime.h>
#include <numa.h>

#include <memory>
#include <vector>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace moe {

/* ---------------------------------------------------------------------------------------------- */
/*                                          GPU Alloc                                             */
/* ---------------------------------------------------------------------------------------------- */
void gpuFreeHost(void* ptr);
void* gpuCallocHost(size_t bytes, unsigned int flags);

template <class T = void>
struct GpuHostDeleter {
  void operator()(void* ptr) { gpuFreeHost(ptr); }
};

template <class T>
using UniqueGpuHostPtr = std::unique_ptr<T, GpuHostDeleter<T>>;

template <class T, class Deleter, class Memory, typename Alloc, typename... Args>
Memory safeAlloc(Alloc alloc, size_t nelems, Args&&... args) {
  T* ptr = nullptr;
  try {
    ptr = reinterpret_cast<T*>(alloc(nelems * sizeof(T), std::forward<Args>(args)...));
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

template <class T>
auto gpuCallocHostUnique(size_t nelems = 1, unsigned int flags = hipHostAllocMapped) {
  return safeAlloc<T, GpuHostDeleter<T>, UniqueGpuHostPtr<T>>(gpuCallocHost, nelems, flags);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            NUMA                                                */
/* ---------------------------------------------------------------------------------------------- */
static const std::string getBusId(int deviceId);
int getDeviceNumaNode(int deviceId);
void numaBind(int node);

/* ---------------------------------------------------------------------------------------------- */
/*                                          Stream Pool                                           */
/* ---------------------------------------------------------------------------------------------- */
class StreamPool {
 public:
  StreamPool(int npes);
  ~StreamPool();
  hipStream_t GetStream(int destPe);

 private:
  std::vector<hipStream_t> streams_;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                          Atomic                                                */
/* ---------------------------------------------------------------------------------------------- */

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

// HIP does not have thread scope enums like CUDA
constexpr auto scopeSystem = 0;
constexpr auto scopeDevice = 0;

template <typename T, int scope = scopeSystem>
T AtomicLoad(const T* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T, int scope = scopeSystem>
void AtomicStore(T* ptr, const T& val, int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T, int scope = scopeSystem>
T AtomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

}  // namespace moe
}  // namespace mori
