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

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

inline __device__ index_t FindSrcPe(const index_t* ps, index_t tokenIdx, int worldSize) {
  index_t srcPe = 0;
  while (ps[srcPe] <= tokenIdx) {
    srcPe++;
  }
  return srcPe;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void SetCrossDeviceBarrierAndSignalProxy(EpDispatchCombineArgs<T> args,
                                                           const uint32_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  index_t* recvTokenNumObj = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalThdId < args.config.worldSize) {
    // Set remote flag after all copies are done
    if (globalThdId == 0) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
      args.combineGridBarrier[0] = 0;
      core::AtomicStoreRelaxedSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>() + args.config.rank,
          crossDeviceBarrierFlag);
    }

    // Signal Host proxy
    index_t recvTokenNum = core::AtomicLoadRelaxed(recvTokenNumObj + globalThdId);
    assert(recvTokenNum >= 0);
    args.hostTokenCounts[globalThdId] = recvTokenNum;
    __threadfence_system();

    if (globalThdId == 0) {
      args.proxyTrigger->SetEvent(EventBitFlag::TriggerCombine);
    }
  }
}

template <typename T>
inline __device__ void WaitCrossDeviceBarrier(EpDispatchCombineArgs<T> args,
                                              const uint32_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int npes = args.config.worldSize;

  // Wait cross device barrier
  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < npes) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchIntraNodeOverlapSendKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  size_t weightOffset = config.hiddenDim * sizeof(T);
  size_t indicesOffset = weightOffset + sizeof(float) * config.numExpertPerToken;
  size_t scalesOffset = indicesOffset + sizeof(index_t) * config.numExpertPerToken;
  size_t stagingOffset = scalesOffset + config.scaleTypeSize * config.scaleDim;

  size_t maxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();

  if (args.tokenIndices && args.inpTokenBuf) {
    // Phase1: send token
    // Each warp compute token offset on destinition PE
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
         i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t destTokId = 0;

      // Deduplicate
      assert(config.numExpertPerToken < warpSize);
      int condition = 0;
      if (laneId < (i % config.numExpertPerToken)) {
        condition = destPe == (args.tokenIndices[srcTokId * config.numExpertPerToken + laneId] /
                               config.numExpertPerRank);
      }
      if (__any(condition)) {
        if (laneId == 0) args.dispDestTokIdMap[i] = config.worldSize * maxNumTokensToSendPerRank;
        continue;
      }

      if (laneId == 0) {
        // decide token id in dest pe
        destTokId = atomicAdd(args.destPeTokenCounter + destPe, 1);
        args.dispDestTokIdMap[i] = destPe * maxNumTokensToSendPerRank + destTokId;
      }
      destTokId = __shfl(destTokId, 0);
      index_t destTokOffset = (destPe * maxNumTokensToSendPerRank + destTokId) * stagingOffset;
      index_t srcTokOffset = srcTokId * size_t(config.hiddenDim) * sizeof(T);

      // Copy hidden states
      core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + destTokOffset,
                     reinterpret_cast<char*>(args.inpTokenBuf) + srcTokOffset,
                     config.hiddenDim * sizeof(T));
      // Copy topk weights if exist
      if (args.weightsBuf) {
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + destTokOffset + weightOffset,
            reinterpret_cast<char*>(args.weightsBuf) +
                srcTokId * config.numExpertPerToken * sizeof(float),
            config.numExpertPerToken * sizeof(float));
      }
      // Copy topk ids
      core::WarpCopy(
          args.shmemStagingTokMemObj->template GetAs<char*>() + destTokOffset + indicesOffset,
          reinterpret_cast<char*>(args.tokenIndices) +
              srcTokId * config.numExpertPerToken * sizeof(index_t),
          config.numExpertPerToken * sizeof(index_t));
      // Copy scales if exist
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        index_t srcScaleOffset = srcTokId * config.scaleDim * config.scaleTypeSize;
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + destTokOffset + scalesOffset,
            reinterpret_cast<char*>(args.scalesBuf) + srcScaleOffset,
            config.scaleDim * config.scaleTypeSize);
      }
    }
  }
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
      args.dispatchGridBarrier[0] = 0;
      index_t numToken = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
      args.destPeTokenCounter[destPe] = 0;
      core::AtomicStoreRelaxedSystem(args.sendTokenNumMemObj->template GetAs<index_t*>() + destPe,
                                     numToken);
      // Store token counts to host buffer
      args.hostTokenCounts[destPe] = numToken;
    }
    __threadfence_system();
    // send signal to cpu proxy
    if (laneId == 0) {
      uint8_t val = 1;
      core::AtomicStoreRelaxedSystem(args.sendAtomicSignalMemObj->template GetAs<uint8_t*>() + myPe,
                                     val);
      args.proxyTrigger->SetEvent(EventBitFlag::TriggerDispatch);
    }
  }
}

template <typename T>
__global__ void EpDispatchIntraNodeOverlapRecvKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  size_t weightOffset = config.hiddenDim * sizeof(T);
  size_t indicesOffset = weightOffset + sizeof(float) * config.numExpertPerToken;
  size_t scalesOffset = indicesOffset + sizeof(index_t) * config.numExpertPerToken;
  size_t stagingOffset = scalesOffset + config.scaleTypeSize * config.scaleDim;

  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  extern __shared__ index_t recvTokenNums[];
  if (warpId == 0) {
    // Get num tokens
    index_t* recvTokenNumObj = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      shmem::ShmemUint8WaitUntilGreaterThan(
          args.sendAtomicSignalMemObj->template GetAs<uint8_t*>() + srcPe, 0);
      index_t recvTokenNum = core::AtomicLoadRelaxedSystem(recvTokenNumObj + srcPe);
      recvTokenNums[srcPe] = recvTokenNum;
    }
    if (laneId == 0) {
      for (int i = 1; i < npes; ++i) {
        recvTokenNums[i] += recvTokenNums[i - 1];
      }
    }
  }
  __syncthreads();

  index_t totalTokenNum = recvTokenNums[npes - 1];
  size_t maxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  for (int tokenIdx = globalWarpId; tokenIdx < totalTokenNum; tokenIdx += globalWarpNum) {
    int srcPe = 0;
    if (laneId == 0) {
      srcPe = FindSrcPe(recvTokenNums, tokenIdx, npes);
    }
    srcPe = __shfl(srcPe, 0);
    index_t srcTokenIdx = tokenIdx - (srcPe == 0 ? 0 : recvTokenNums[srcPe - 1]);

    size_t localTokenOffset = tokenIdx * size_t(config.hiddenDim) * sizeof(T);
    size_t srcTokenOffset = (srcPe * MaxNumTokensToRecvPerRank + srcTokenIdx) * stagingOffset;

    core::WarpCopy(args.shmemDispatchOutTokMemObj->template GetAs<char*>() + localTokenOffset,
                   args.shmemDispatchInpTokMemObj->template GetAs<char*>() + srcTokenOffset,
                   config.hiddenDim * sizeof(T));
    core::WarpCopy(
        args.shmemDispatchOutWeightsMemObj->template GetAs<char*>() +
            tokenIdx * config.numExpertPerToken * sizeof(float),
        args.shmemDispatchInpTokMemObj->template GetAs<char*>() + srcTokenOffset + weightOffset,
        config.numExpertPerToken * sizeof(float));
    core::WarpCopy(
        args.shmemOutIndicesMemObj->template GetAs<char*>() +
            tokenIdx * config.numExpertPerToken * sizeof(index_t),
        args.shmemDispatchInpTokMemObj->template GetAs<char*>() + srcTokenOffset + indicesOffset,
        config.numExpertPerToken * sizeof(index_t));
    if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
      core::WarpCopy(
          args.shmemOutScalesMemObj->template GetAs<char*>() +
              tokenIdx * config.scaleDim * config.scaleTypeSize,
          args.shmemDispatchInpTokMemObj->template GetAs<char*>() + srcTokenOffset + scalesOffset,
          config.scaleDim * config.scaleTypeSize);
    }
  }

  if (globalThdId == 0) {
    args.totalRecvTokenNum[0] = totalTokenNum;
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineIntraNodeOverlapSendKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];

  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];

  size_t weightSize = args.weightsBuf ? sizeof(float) * config.numExpertPerToken : 0;
  size_t hiddenSize = config.hiddenDim * sizeof(T);
  size_t stagingOffset = weightSize + hiddenSize;

  int npes = config.worldSize;

  for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
    size_t destTokOffset = i * stagingOffset;
    size_t srcTokOffset = i * hiddenSize;

    core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + destTokOffset,
                   reinterpret_cast<char*>(args.inpTokenBuf) + srcTokOffset, hiddenSize);

    if (args.weightsBuf) {
      size_t srcWeightOffset = i * weightSize;
      core::WarpCopy(
          args.shmemStagingTokMemObj->template GetAs<char*>() + destTokOffset + hiddenSize,
          reinterpret_cast<char*>(args.weightsBuf) + srcWeightOffset, weightSize);
    }
  }

  SetCrossDeviceBarrierAndSignalProxy(args, crossDeviceBarrierFlag);
  *args.totalRecvTokenNum = 0;
  if (globalThdId < npes) {
    core::AtomicStoreRelaxedSystem(
        args.sendTokenNumMemObj->template GetAs<index_t*>() + globalThdId, 0);
    core::AtomicStoreRelaxedSystem(
        args.recvTokenNumMemObj->template GetAs<index_t*>() + globalThdId, 0);
    core::AtomicStoreRelaxedSystem(
        args.sendAtomicSignalMemObj->template GetAs<uint8_t*>() + globalThdId, (uint8_t)0);
  }
}

template <typename T>
__global__ void EpCombineIntraNodeOverlapRecvKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];

  int myPe = config.rank;
  int npes = config.worldSize;
  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  size_t weightSize = args.weightsBuf ? sizeof(float) * config.numExpertPerToken : 0;
  size_t hiddenSize = config.hiddenDim * sizeof(T);
  size_t stagingOffset = weightSize + hiddenSize;

  size_t maxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();

  // Wait cross device barrier
  WaitCrossDeviceBarrier(args, crossDeviceBarrierFlag);
  assert(config.numExpertPerToken < warpSize);

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    // Prepare data pointers on different GPUs
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = destTokId / maxNumTokensToSendPerRank;
      index_t destTokOffset = destTokId * stagingOffset;

      if (destPe < config.worldSize) {
        index_t destLocalTokId = destTokId - destPe * maxNumTokensToSendPerRank;
        srcPtrs[j] = reinterpret_cast<T*>(args.shmemCombineInpTokMemObj->template GetAs<char*>() +
                                          destTokOffset + hiddenDimOffset * sizeof(T));
        srcWeightsPtr[j] = reinterpret_cast<float*>(
            args.shmemCombineInpTokMemObj->template GetAs<char*>() + destTokOffset + hiddenSize);
      } else {
        srcPtrs[j] = nullptr;
        srcWeightsPtr[j] = nullptr;
      }
    }
    core::WarpAccum<T, 4>(args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                              tokenId * config.hiddenDim + hiddenDimOffset,
                          srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);

    if (args.weightsBuf && inTokenPartId == warpsPerToken - 1) {
      core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtr, nullptr, config.numExpertPerToken,
                                config.numExpertPerToken);
    }
  }
  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId == 0) {
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
    args.combineGridBarrier[0] = 0;
    __hip_atomic_fetch_add(args.crossDeviceBarrierFlag, 1, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

}  // namespace moe
}  // namespace mori
