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

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mori {
namespace moe {

enum class DmaTransferStatus { SUCCESS = 0, ERROR = 1 };

struct TransferTask {
  void* srcMem;
  void* dstMem;
  size_t size;
  hsa_agent_t srcAgent;
  hsa_agent_t dstAgent;
  int srcDevice;
  int dstDevice;
  bool needSync;
};

using TransferTaskList = std::vector<std::unique_ptr<TransferTask>>;

class HsaSignal {
 public:
  HsaSignal(int deviceIdx);
  ~HsaSignal();

  hsa_signal_t& Get() { return signal; }

 private:
  int deviceIdx_;
  hsa_signal_t signal;
};

class HsaSignalPool {
 public:
  using DeviceIdx = int;
  HsaSignalPool(int numSignalPerDevice, int npes);

  std::vector<hsa_signal_t> GetWaitSignals(int deviceIdx);
  hsa_signal_t& GetFreeSignal(int deviceIdx, hsa_signal_value_t initVal);
  void Cleanup(int deviceIdx);

 private:
  int npes_;
  std::unordered_map<DeviceIdx, std::deque<std::unique_ptr<HsaSignal>>> freeSignals;
  std::unordered_map<DeviceIdx, std::deque<std::unique_ptr<HsaSignal>>> activeSignals;
};

class DmaTransferEngine {
 public:
  DmaTransferEngine(int npes);
  ~DmaTransferEngine() = default;

  std::unique_ptr<TransferTask> CreateTransferTask(void* dst, void* src, size_t size,
                                                   int srcDeviceIdx, int dstDeviceIdx,
                                                   bool needSync = false);
  void ExecuteDmaTransfer(TransferTaskList& transferTasks);
  void Cleanup() {
    for (int i = 0; i < npes_; ++i) {
      signalPool.Cleanup(i);
    }
  }

 private:
  int npes_;
  HsaSignalPool signalPool;
};

}  // namespace moe
}  // namespace mori
