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
#include "src/ops/dispatch_combine/internode.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierInterNodeKernelForDivided(EpDispatchCombineArgs<T> args, int loop) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalThdId / warpSize;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  // TODO: still figure out why use multiple threads lost RDMA writes
  for (int destPe = globalWarpId; destPe < args.config.worldSize; destPe += globalWarpNum) {
    if (laneId == 0) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
      shmem::ShmemPutUint32ImmNbiWarp(args.crossDeviceBarrierMemObj,
                                      args.config.rank * sizeof(uint32_t),
                                      args.crossDeviceBarrierFlag+loop, destPe);
    }
  }

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != args.crossDeviceBarrierFlag+loop) {
    }
  }
  __syncthreads();
}


/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernelDivided                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeKernelDivided(EpDispatchCombineArgs<T> args, int gpu_per_node, float clockRateInKHz, int loop) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int myPe = config.rank;
  int npes = config.worldSize;
  int myNode = myPe / gpu_per_node;

  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  // Phase 1: send token
  // This phase is symmetric with dispatch recv phase, where tokens are first sent back to its
  // source pe in pe sorted order

  /** SUCCESS **/
  const int half_npes = (npes + 1) / 2;
  const int numsBlockPerSrcPe = gridDim.x / half_npes;
  const int srcPe = (blockIdx.x / numsBlockPerSrcPe) * 2 + loop;

  const int srcNode = srcPe / gpu_per_node;
  const int localBlockId = blockIdx.x % numsBlockPerSrcPe;
  const int srcPeTokenNum = *(args.recvTokenNumMemObj->template GetAs<index_t*>() + srcPe +
                              (args.crossDeviceBarrierFlag & 1) * npes) -
                            1;
  const int baseChunk = srcPeTokenNum / numsBlockPerSrcPe;
  const int remainder = srcPeTokenNum % numsBlockPerSrcPe;

  const int myChunkSize = baseChunk + (localBlockId < remainder);

  const int startIdx = localBlockId * baseChunk + min(localBlockId, remainder);
  const int endIdx = startIdx + myChunkSize;

  const size_t tokenSize = config.hiddenDim * sizeof(T);
  const size_t weightSize = args.weightsBuf ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t tokenPackSize = tokenSize + weightSize;

  if (srcNode == myNode) {
    // intra node use xgmi for transfer
    for (int idx = warpId; idx < endIdx - startIdx; idx += warpNum) {
      const index_t mapIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t mapIdxOffset = mapIdx * tokenPackSize;
      const index_t tokenId = args.srcPeTokenIdxMap[mapIdx];
      size_t tokenOffset = tokenId * tokenSize;
      const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t peSortedOffset = peSortedId * tokenPackSize;
      core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                     reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);

      if (args.weightsBuf) {
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + tokenSize,
            reinterpret_cast<char*>(args.weightsBuf) +
                tokenId * config.numExpertPerToken * sizeof(float),
            weightSize);
      }

      shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, peSortedOffset,
                                          args.shmemStagingTokMemObj, mapIdxOffset, tokenPackSize,
                                          srcPe);
    }
  } else {
    // inter node use ibgda for transfer
    // last warp for coordinate, other warp for gather token
    __shared__ int gatherTokenNum[1024];
    for (int idx = thdId; idx < 1024; idx += thdNum) {
      gatherTokenNum[idx] = 0;
    }
    __syncthreads();
    const int chunkTokenSize = (warpNum - 1);
    if (warpId == warpNum - 1) {
      const int totalTokenInBlock = endIdx - startIdx;
      int chunkOffset = 0;
      int chunkIdx = 0;
      while (chunkOffset < totalTokenInBlock) {
        int actualTokenNum = totalTokenInBlock - chunkOffset < chunkTokenSize
                                 ? totalTokenInBlock - chunkOffset
                                 : chunkTokenSize;
        if (laneId == 0) {
          while (atomicAdd(&gatherTokenNum[chunkIdx], 0) < actualTokenNum) {
            ;
          }
        }
        // rdma_send
        const index_t srcIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t srcOffset = srcIdx * tokenPackSize;
        const index_t dstIdx = myPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t dstOffset = dstIdx * tokenPackSize;
        shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, dstOffset,
                                            args.shmemStagingTokMemObj, srcOffset,
                                            actualTokenNum * tokenPackSize, srcPe);

        ++chunkIdx;
        chunkOffset += chunkTokenSize;
      }
    } else {
      // int warpTokens = 0;
      int chunkIdx = 0;
      for (int idx = warpId; idx < endIdx - startIdx; idx += chunkTokenSize) {
        const index_t mapIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        size_t mapIdxOffset = mapIdx * tokenPackSize;
        const index_t tokenId = args.srcPeTokenIdxMap[mapIdx];
        size_t tokenOffset = tokenId * tokenSize;
        // const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        // size_t peSortedOffset = peSortedId * size_t(config.hiddenDim);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                       reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);

        if (args.weightsBuf) {
          core::WarpCopy(
              args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + tokenSize,
              reinterpret_cast<char*>(args.weightsBuf) +
                  tokenId * config.numExpertPerToken * sizeof(float),
              weightSize);
        }
        if (laneId == 0) atomicAdd(&gatherTokenNum[chunkIdx++], 1);
      }
      // if (laneId == 0 && warpTokens) atomicAdd(&gatherTokenNum, warpTokens);
      __threadfence_block();
    }
  }
  SyncIfDebugEnabled("Combine kernel: send token end");

  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierInterNodeKernelForDivided(args, loop);
  shmem::ShmemQuietThread();

  if (globalThdId == 0) {
    __hip_atomic_store(args.combineGridBarrier, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}
/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernelFinalize                            */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeKernelFinalize(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int npes = config.worldSize;

  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  const size_t tokenSize = config.hiddenDim * sizeof(T);
  const size_t weightSize = args.weightsBuf ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t tokenPackSize = tokenSize + weightSize;

  if (globalThdId < npes) {
    args.recvTokenNumMemObj
        ->template GetAs<index_t*>()[globalThdId + (args.crossDeviceBarrierFlag & 1) * npes] = 0;
  }

  if (globalThdId == 0) {
    args.localPeTokenCounter[0] = 0;
    args.totalRecvTokenNum[0] = 0;
  }

  SyncIfDebugEnabled("Dispatch kernel: sync across device end");

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;

  int warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  size_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    int tokenId = i / warpsPerToken;
    int inTokenPartId = i % warpsPerToken;
    size_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    size_t hiddenDimSize = std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp);

    // Prepare data pointers on different GPUs
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t peSortedId = args.dispSenderIdxMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = peSortedId / MaxNumTokensToRecvPerRank;
      size_t byteOffset = size_t(peSortedId) * tokenPackSize + hiddenDimOffset * sizeof(T);
      size_t weightByteOffset = size_t(peSortedId) * tokenPackSize + tokenSize;

      if (destPe < config.worldSize) {
        srcPtrs[j] =
            reinterpret_cast<T*>(args.shmemInpTokMemObj->template GetAs<char*>() + byteOffset);
        srcWeightsPtr[j] = reinterpret_cast<float*>(
            args.shmemInpTokMemObj->template GetAs<char*>() + weightByteOffset);
      } else {
        srcPtrs[j] = nullptr;
        srcWeightsPtr[j] = nullptr;
      }
    }

    size_t offset = size_t(tokenId) * size_t(config.hiddenDim) + hiddenDimOffset;
    core::WarpAccum<T, 8>(args.shmemOutTokMemObj->template GetAs<T*>() + offset, srcPtrs, nullptr,
                          config.numExpertPerToken, hiddenDimSize);

    if (args.weightsBuf && inTokenPartId == warpsPerToken - 1) {
      core::WarpAccum<float, 4>(
          args.shmemOutWeightsMemObj->template GetAs<float*>() + tokenId * config.numExpertPerToken,
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

}  // namespace moe
}  // namespace mori

