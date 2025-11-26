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

#include "mori/ops/dispatch_combine/proxy.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <thread>
#include <vector>

#include "mori/ops/dispatch_combine/common.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/ops/dispatch_combine/utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace {

size_t GetTokenSize(const mori::moe::EpDispatchCombineConfig& config, size_t dtypeSize) {
  size_t inputSize = config.hiddenDim * dtypeSize;
  size_t weightSize = sizeof(float) * config.numExpertPerToken;
  size_t indicesSize = sizeof(mori::moe::index_t) * config.numExpertPerToken;
  size_t scalesSize = config.scaleTypeSize * config.scaleDim;
  return inputSize + weightSize + indicesSize + scalesSize;
}

template <typename T>
std::vector<T> CumulativeSum(T* arr, size_t nelems) {
  std::vector<T> cumsum(nelems);
  T sum = 0;
  std::transform(arr, arr + nelems, cumsum.begin(), [&sum](T x) { return sum += x; });
  return cumsum;
}

}  // namespace

namespace mori {
namespace moe {

constexpr int MaxNumGpuPerNode = 8;
constexpr int ProxyStopCheckPeriod = 1000;
constexpr int ProxyStartWarnPeriod = 1000;

/* ---------------------------------------------------------------------------------------------- */
/*                                          Proxy                                                 */
/* ---------------------------------------------------------------------------------------------- */
Proxy::Proxy(const EpDispatchCombineHandle& handle)
    : handle_(handle),
      eventManager(std::make_unique<EventManager>()),
      running(false),
      threadStarted(false),
      hostTokenCounts(gpuCallocHostUnique<index_t>(handle.config.worldSize)),
      hostSignal(gpuCallocHostUnique<uint8_t>()),
      streamPool(handle.config.worldSize) {
  AtomicStore(hostSignal.get(), (uint8_t)1, memoryOrderAcquire);
}

Proxy::~Proxy() {
  if (IsRunning()) {
    Stop();
  }
}

void Proxy::Start() {
  running.store(true, std::memory_order_release);

  int deviceId;
  HIP_RUNTIME_CHECK(hipGetDevice(&deviceId));
  auto initThread = [deviceId]() {
    HIP_RUNTIME_CHECK(hipSetDevice(deviceId));
    int deviceNumaNode = getDeviceNumaNode(deviceId);
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
      MORI_OPS_INFO("NUMA node of proxy thread (device {}) is set to {}", deviceId, deviceNumaNode);
    }
  };

  service = std::thread([this, initThread] {
    initThread();
    threadStarted.store(true, std::memory_order_release);
    MORI_OPS_INFO("Proxy thread has started");

    int runCnt = 0;
    while (true) {
      if (runCnt++ == ProxyStopCheckPeriod) {
        runCnt = 0;
        if (!IsRunning()) {
          break;
        }
      }
      auto event = eventManager->Poll();
      if (!IsAnyEventSet(event, EventBitFlag::TriggerDispatch | EventBitFlag::TriggerCombine)) {
        continue;
      }

      if (IsEventSet(event, EventBitFlag::TriggerDispatch)) {
        TriggerDispatch();
      } else {
        assert(event == EventBitFlag::TriggerCombine);
        TriggerCombine();
      }
    }
  });

  int count = 0;
  while (!threadStarted.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (count++ == ProxyStartWarnPeriod) {
      count = 0;
      MORI_OPS_WARN("Proxy thread startup taking longer than expected.");
    }
  }
}

void Proxy::Stop() {
  running.store(false, std::memory_order_release);
  if (service.joinable()) {
    service.join();
  }
}

void Proxy::TriggerDispatch() {
  const EpDispatchCombineConfig& config = handle_.config;
  size_t tokenSize = GetTokenSize(config, GetHipDataTypeSize(handle_.inputType));
  size_t maxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t maxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();
  int myPe = config.rank;
  int npes = config.worldSize;
  MORI_OPS_INFO("[{}] Proxy::TriggerDispatch", myPe);

  // Clear host token counts
  std::vector<index_t> destTokenCounts(npes);
  std::copy(hostTokenCounts.get(), hostTokenCounts.get() + npes, destTokenCounts.begin());
  ::memset(hostTokenCounts.get(), 0, npes * sizeof(index_t));

  for (int destPe = 0; destPe < npes; ++destPe) {
    auto stream = streamPool.GetStream(destPe);
    index_t destTokenCount = destTokenCounts[destPe];

    // Send hidden states + weights + indices + scales to remote rank
    size_t srcOffset = destPe * (maxNumTokensToSendPerRank * tokenSize);
    size_t destPeOffset = myPe * (maxNumTokensToRecvPerRank * tokenSize);
    void* src = handle_.shmemStagingTokMemObj->GetAs<char*>() + srcOffset;
    void* dst = handle_.shmemDispatchInpTokMemObj->GetAs<char*>(destPe) + destPeOffset;
    size_t nbytes = tokenSize * destTokenCount;
    HIP_RUNTIME_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDeviceToDeviceNoCU, stream));

    // Send send token count to remote rank
    void* tokenNumSrc = handle_.sendTokenNumMemObj->GetAs<index_t*>() + destPe;
    void* tokenNumDst = handle_.recvTokenNumMemObj->GetAs<index_t*>(destPe) + myPe;
    HIP_RUNTIME_CHECK(hipMemcpyAsync(tokenNumDst, tokenNumSrc, sizeof(index_t),
                                     hipMemcpyDeviceToDeviceNoCU, stream));
    // Send Signal
    void* signalDst = handle_.sendAtomicSignalMemObj->GetAs<uint8_t*>(destPe) + myPe;
    HIP_RUNTIME_CHECK(hipMemcpyAsync(signalDst, hostSignal.get(), sizeof(uint8_t),
                                     hipMemcpyHostToDevice, stream));
  }
  //::memset(hostTokenCounts.get(), 0, npes * sizeof(index_t));
  // Clear host token counts
}

void Proxy::TriggerCombine() {
  const EpDispatchCombineConfig& config = handle_.config;
  int myPe = config.rank;
  int npes = config.worldSize;
  size_t maxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  MORI_OPS_INFO("[{}] Proxy::TriggerCombine", myPe);

  size_t weightSize = handle_.weightsBuf ? sizeof(float) * config.numExpertPerToken : 0;
  size_t hiddenSize = config.hiddenDim * GetHipDataTypeSize(handle_.inputType);
  size_t stagingSize = hiddenSize + weightSize;
  auto tokenCountPrefixSum = CumulativeSum(hostTokenCounts.get(), npes);
  ::memset(hostTokenCounts.get(), 0, npes * sizeof(index_t));

  for (int destPe = 0; destPe < npes; ++destPe) {
    auto stream = streamPool.GetStream(destPe);

    size_t destTokenStartIdx = destPe == 0 ? 0 : tokenCountPrefixSum[destPe - 1];
    size_t destTokenCount = tokenCountPrefixSum[destPe] - destTokenStartIdx;

    size_t srcOffset = destTokenStartIdx * stagingSize;
    size_t destOffset = myPe * (maxNumTokensToSendPerRank * stagingSize);
    void* src = handle_.shmemStagingTokMemObj->GetAs<char*>() + srcOffset;
    void* dst = handle_.shmemCombineInpTokMemObj->GetAs<char*>(destPe) + destOffset;
    size_t nbytes = stagingSize * destTokenCount;
    HIP_RUNTIME_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDeviceToDeviceNoCU, stream));

    // Send cross device barrier to dest rank
    void* srcBarrier = handle_.crossDeviceBarrierMemObj->Get();
    void* dstBarrier = handle_.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(destPe) + myPe;
    HIP_RUNTIME_CHECK(
        hipMemcpyAsync(dstBarrier, srcBarrier, sizeof(uint32_t), hipMemcpyHostToDevice, stream));
  }
}

bool Proxy::IsRunning() { return running.load(std::memory_order_acquire); }

/* ---------------------------------------------------------------------------------------------- */
/*                                          EventManager                                          */
/* ---------------------------------------------------------------------------------------------- */
EventManager::EventManager() : proxyTrigger(gpuCallocHostUnique<ProxyTrigger>()) {}

EventBitFlag EventManager::Poll() {
  EventBitFlag event = proxyTrigger->GetEvent();
  if (event != EventBitFlag::None) {
    proxyTrigger->ClearEvent();
  }
  return event;
}

}  // namespace moe
}  // namespace mori
