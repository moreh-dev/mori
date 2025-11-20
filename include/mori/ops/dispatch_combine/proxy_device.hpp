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

#include "mori/core/utils.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/ops/dispatch_combine/utils.hpp"

namespace mori {
namespace moe {

using EventBitFlagType = uint32_t;
enum class EventBitFlag : EventBitFlagType {
  None = 0x0,
  TriggerDispatch = 0x1,
  TriggerCombine = 0x2
};

inline EventBitFlag operator&(const EventBitFlag& lhs, const EventBitFlag& rhs) {
  return static_cast<EventBitFlag>(static_cast<EventBitFlagType>(lhs) &
                                   static_cast<EventBitFlagType>(rhs));
}

inline EventBitFlag operator|(const EventBitFlag& lhs, const EventBitFlag& rhs) {
  return static_cast<EventBitFlag>(static_cast<EventBitFlagType>(lhs) |
                                   static_cast<EventBitFlagType>(rhs));
}

inline bool operator==(const EventBitFlag& lhs, const EventBitFlag& rhs) {
  return static_cast<EventBitFlagType>(lhs) == static_cast<EventBitFlagType>(rhs);
}

inline bool operator!=(const EventBitFlag& lhs, const EventBitFlag& rhs) {
  return static_cast<EventBitFlagType>(lhs) != static_cast<EventBitFlagType>(rhs);
}

inline bool IsEventSet(const EventBitFlag& self, const EventBitFlag& event) {
  return (self & event) == event;
}

inline bool IsAnyEventSet(const EventBitFlag& self, const EventBitFlag& event) {
  return (self & event) != EventBitFlag::None;
}

struct ProxyTrigger {
  ProxyTrigger() : event(EventBitFlag::None) {}

  __host__ __forceinline__ EventBitFlag GetEvent() {
    return static_cast<EventBitFlag>(AtomicLoad<EventBitFlagType>(
        reinterpret_cast<EventBitFlagType*>(&proxyEvent), memoryOrderAcquire));
  }

  __device__ __forceinline__ void SetEvent(EventBitFlag event) {
    core::AtomicStoreRelaxedSystem(reinterpret_cast<EventBitFlagType*>(&proxyEvent),
                                   static_cast<EventBitFlagType>(event));
  }

  __host__ __forceinline__ void ClearEvent() {
    AtomicStore<EventBitFlagType>(
        reinterpret_cast<EventBitFlagType*>(&(proxyTrigger.get()->proxyEvent)), memoryOrderRelease);
  }

  EventBitFlag proxyEvent;
};

}  // namespace moe
}  // namespace mori
