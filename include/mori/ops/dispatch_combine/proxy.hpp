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

#include "mori/ops/dispatch_combine/common.hpp"
#include "mori/ops/dispatch_combine/proxy_device.hpp"
#include "mori/ops/dispatch_combine/utils.hpp"

namespace mori {
namespace moe {

class EpDispatchCombineHandle;
class EventManager;

// Host-side proxy for dispatch/combine intiation
class Proxy {
 public:
  Proxy(const EpDispatchCombineHandle& handle);
  ~Proxy();

  void Start();
  void Stop();
  bool IsRunning();
  EventManager* GetEventManager() { return eventManager.get(); }
  index_t* GetHostTokenCounts() { return hostTokenCounts.get(); }

 private:
  void TriggerDispatch();
  void TriggerCombine();

  const EpDispatchCombineHandle& handle_;
  std::unique_ptr<EventManager> eventManager;
  std::atomic_bool running;
  std::atomic_bool threadStarted;
  std::thread service;
  UniqueGpuHostPtr<index_t> hostTokenCounts;
  StreamPool streamPool;
};

class EventManager {
 public:
  EventManager();
  ProxyTrigger* GetProxyTrigger() { return proxyTrigger.get(); }
  EventBitFlag Poll();

 private:
  UniqueGpuHostPtr<ProxyTrigger> proxyTrigger;
};

}  // namespace moe
}  // namespace mori
