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

  size_t maxNumTokensToSend = config.MaxNumTokensToSend();

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
        if (laneId == 0) args.dispDestTokIdMap[i] = config.worldSize * maxNumTokensToSend;
        continue;
      }

      if (laneId == 0) {
        // decide token id in dest pe
        destTokId = atomicAdd(args.destPeTokenCounter + destPe, 1);
        args.dispDestTokIdMap[i] = destPe * maxNumTokensToSend + destTokId;
      }
      destTokId = __shfl(destTokId, 0);
      index_t destTokOffset = (destPe * maxNumTokensToSend + destTokId) * stagingOffset;
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
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      core::AtomicStoreRelaxedSystem(args.sendTokenNumMemObj->template GetAs<index_t*>() + destPe,
                                     numTokenSignal);
      args.destPeTokenCounter[destPe] = 0;
      // Store token counts to host buffer
      args.hostTokenCounts[destPe] = numTokenSignal - 1;
    }
    __threadfence_system();
    // send signal to cpu proxy
    if (laneId == 0) {
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

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  size_t weightOffset = config.hiddenDim * sizeof(T);
  size_t indicesOffset = weightOffset + sizeof(float) * config.numExpertPerToken;
  size_t scalesOffset = indicesOffset + sizeof(index_t) * config.numExpertPerToken;
  size_t stagingOffset = scalesOffset + config.scaleTypeSize * config.scaleDim;

  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  __shared__ index_t recvTokenNums[MAX_GPUS_PER_NODE];
  if (warpId == 0) {
    index_t* recvTokenNumObj = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      index_t* signal = recvTokenNumObj + srcPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      recvTokenNums[srcPe] = recvTokenNum;
    }
    if (thdId == 0) {
      for (int i = 1; i < npes; ++i) {
        recvTokenNums[i] += recvTokenNums[i - 1];
      }
    }
  }
  __syncthreads();

  index_t totalTokenNum = recvTokenNums[npes - 1];
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
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Reset signal to 0
  if (globalWarpId == 0) {
    if (size_t destPe = laneId; destPe < npes) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
      args.dispatchGridBarrier[0] = 0;

      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>() + destPe;
      core::AtomicStoreRelaxedSystem(signal, 0);
    }
  }
}

}  // namespace moe
}  // namespace mori
