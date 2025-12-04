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

#include "mori/ops/dispatch_combine/dma_transfer.hpp"

#include <hip/hip_runtime.h>
#include <stdio.h>

#include <future>
#include <iostream>

#include "mori/utils/mori_log.hpp"

namespace mori {
namespace moe {

#define HSA_CHECK(stmt)                                                                      \
  do {                                                                                       \
    hsa_status_t status = (stmt);                                                            \
    if (status != HSA_STATUS_SUCCESS) {                                                      \
      fprintf(stderr, "[%s:%d] hsa failed with status 0x%x \n", __FILE__, __LINE__, status); \
      exit(-1);                                                                              \
    }                                                                                        \
  } while (0)

constexpr int NumSignalPerDevice = 8;
constexpr hsa_signal_value_t InitSignalValueOne = 1;
constexpr int MinAsyncCopyOnEngineSize = 1024 * 1024;
constexpr uint32_t ExcludeMask = 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x1000 | 0x2000;

hsa_agent_t GetHsaAgent(void* ptr) {
  hsa_amd_pointer_info_t info;
  info.size = sizeof(info);
  HSA_CHECK(hsa_amd_pointer_info(ptr, &info, NULL, NULL, NULL));
  return info.agentOwner;
}

inline uint32_t GetSdmaEngineId(uint32_t freeMask) {
  if (freeMask == 0) return 0;
  // Try preferred engines first (high-bandwidth engines)
  uint32_t preferredMask = freeMask & ~ExcludeMask;
  if (preferredMask != 0) {
    return preferredMask & (~preferredMask + 1);  // Extract lowest preferred engine
  }

  // Fall back to non-preferred engines (slower engines)
  return freeMask & (~freeMask + 1);  // Extract lowest available engine
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          HsaSignalPool                                         */
/* ---------------------------------------------------------------------------------------------- */
HsaSignal::HsaSignal(int deviceIdx) : deviceIdx_(deviceIdx) {
  // HSA_CHECK(hsa_signal_create(0, 0, nullptr, &signal));
  HSA_CHECK(hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal));
}

HsaSignal::~HsaSignal() { HSA_CHECK(hsa_signal_destroy(signal)); }

HsaSignalPool::HsaSignalPool(int numSignalPerDevice, int npes) : npes_(npes) {
  for (int deviceIdx = 0; deviceIdx < npes; ++deviceIdx) {
    freeSignals[deviceIdx] = std::deque<std::unique_ptr<HsaSignal>>(numSignalPerDevice);
    for (int i = 0; i < numSignalPerDevice; ++i) {
      freeSignals[deviceIdx][i] = std::make_unique<HsaSignal>(deviceIdx);
    }
  }
}

hsa_signal_t& HsaSignalPool::GetFreeSignal(int deviceIdx, hsa_signal_value_t initVal) {
  if (freeSignals[deviceIdx].empty()) Cleanup(deviceIdx);
  auto freeSignal = std::move(freeSignals[deviceIdx].front());
  freeSignals[deviceIdx].pop_front();

  hsa_signal_t& signal = freeSignal->Get();
  hsa_signal_store_relaxed(signal, initVal);
  activeSignals[deviceIdx].push_back(std::move(freeSignal));
  return signal;
}

void HsaSignalPool::Cleanup(int deviceIdx) {
  auto& deviceActiveSignals = activeSignals[deviceIdx];
  while (freeSignals[deviceIdx].empty()) {
    for (auto it = deviceActiveSignals.begin(); it != deviceActiveSignals.end();) {
      assert(*it != nullptr);
      hsa_signal_t& signal = (*it)->Get();
      if (hsa_signal_load_relaxed(signal) == 0) {
        freeSignals[deviceIdx].push_back(std::move(*it));
        it = deviceActiveSignals.erase(it);
      }
    }
  }
}

std::vector<hsa_signal_t> HsaSignalPool::GetWaitSignals(DeviceIdx deviceIdx) {
  std::vector<hsa_signal_t> waitSignals;
  auto& deviceActiveSignals = activeSignals[deviceIdx];

  for (auto it = deviceActiveSignals.begin(); it != deviceActiveSignals.end();) {
    assert(*it != nullptr);
    hsa_signal_t& signal = (*it)->Get();
    if (hsa_signal_load_relaxed(signal) > 0) {
      waitSignals.push_back(signal);
      ++it;
    } else {
      freeSignals[deviceIdx].push_back(std::move(*it));
      it = deviceActiveSignals.erase(it);
    }
  }
  return waitSignals;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          DmaTransferEngine                                     */
/* ---------------------------------------------------------------------------------------------- */
DmaTransferEngine::DmaTransferEngine(int npes)
    : npes_(npes), signalPool(NumSignalPerDevice, npes) {}

std::unique_ptr<TransferTask> DmaTransferEngine::CreateTransferTask(void* dst, void* src,
                                                                    size_t size, int srcDeviceIdx,
                                                                    int dstDeviceIdx,
                                                                    bool needSync) {
  hsa_status_t status;
  hsa_amd_pointer_info_t info;
  auto transferTask = std::make_unique<TransferTask>();

  transferTask->srcMem = src;
  transferTask->dstMem = dst;
  transferTask->size = size;
  transferTask->srcAgent = GetHsaAgent(src);
  transferTask->dstAgent = GetHsaAgent(dst);
  transferTask->srcDevice = srcDeviceIdx;
  transferTask->dstDevice = dstDeviceIdx;
  transferTask->needSync = needSync;

  return std::move(transferTask);
}

void DmaTransferEngine::ExecuteDmaTransfer(TransferTaskList& transferTasks) {
  for (auto& transferTask : transferTasks) {
    std::vector<hsa_signal_t> waitSignals;
    if (transferTask->needSync) {
      waitSignals = signalPool.GetWaitSignals(transferTask->dstDevice);
    }
    hsa_signal_t activeSignal =
        signalPool.GetFreeSignal(transferTask->dstDevice, InitSignalValueOne);
    size_t waitSignalCount = waitSignals.size();

    uint32_t engineIdMask;
    hsa_status_t status = hsa_amd_memory_copy_engine_status(transferTask->dstAgent,
                                                            transferTask->srcAgent, &engineIdMask);
    if (status == HSA_STATUS_SUCCESS && transferTask->size > MinAsyncCopyOnEngineSize) {
      hsa_amd_sdma_engine_id_t sdmaEngineId =
          static_cast<hsa_amd_sdma_engine_id_t>(GetSdmaEngineId(engineIdMask));

      MORI_OPS_INFO("async_copy_on_engine from {} to {} (sdma_engine_id=0x{:x}, size={})",
                    transferTask->srcDevice, transferTask->dstDevice, sdmaEngineId,
                    transferTask->size);
      HSA_CHECK(hsa_amd_memory_async_copy_on_engine(
          transferTask->dstMem, transferTask->dstAgent, transferTask->srcMem,
          transferTask->srcAgent, transferTask->size, waitSignalCount,
          transferTask->needSync ? waitSignals.data() : NULL, activeSignal, sdmaEngineId, true));
    } else {
      MORI_OPS_INFO("async_copy from {} to {} (size={})", transferTask->srcDevice,
                    transferTask->dstDevice, transferTask->size);
      HSA_CHECK(hsa_amd_memory_async_copy(
          transferTask->dstMem, transferTask->dstAgent, transferTask->srcMem,
          transferTask->srcAgent, transferTask->size, waitSignalCount,
          transferTask->needSync ? waitSignals.data() : NULL, activeSignal));
    }
  }
}

}  // namespace moe
}  // namespace mori
