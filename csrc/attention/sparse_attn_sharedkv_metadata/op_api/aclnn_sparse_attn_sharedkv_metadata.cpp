/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_sparse_attn_sharedkv_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_attn_sharedkv_metadata.h"
#include "l0_sparse_attn_sharedkv_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

// ---------------------------------------------------------------------------
// Temporary debug tracing for the PD-disaggregation failure investigation.
// The kernel-side Compute() has been proven NOT to run on the failing call
// (plog start/end pairs predate the failure), so the failure must be located
// in the aclnn host layer: executor creation, ParamsCheck, Contiguous
// sub-ops, attribute serialization (l0op::SparseAttnSharedkvMetadata), or
// task submission (CommonOpExecutorRun).
//
// Gated by VLLM_ASCEND_SAS_ACLNN_DEBUG: set it only on the P node (eager);
// when unset (the default, e.g. on the D node with graph capture) every
// probe compiles down to a no-op boolean check — zero printing, zero
// iostream traffic, safe for stream/graph capture.
// NOTE: remove before merging.
// ---------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

static bool SasAclnnDebugEnabled()
{
    static const bool enabled = (::getenv("VLLM_ASCEND_SAS_ACLNN_DEBUG") != nullptr);
    return enabled;
}

static uint64_t SasDbgNextSeq()
{
    static std::atomic<uint64_t> s_seq{0};
    return s_seq.fetch_add(1) + 1;
}

static void SasDbgLog(const std::string &stage)
{
    if (!SasAclnnDebugEnabled()) {
        return;
    }
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf;
    (void)localtime_r(&t, &tmBuf);
    std::cout << "[SAS_ACLNN][" << std::put_time(&tmBuf, "%H:%M:%S") << "." << std::setw(3) << std::setfill('0')
              << ms << "][pid=" << getpid() << "][seq=" << SasDbgNextSeq() << "] " << stage << std::endl;
}

#define SAS_DBG(msg)                                      \
    do {                                                  \
        if (SasAclnnDebugEnabled()) {                     \
            std::ostringstream ossSasDbg;                 \
            ossSasDbg << msg;                             \
            SasDbgLog(ossSasDbg.str());                   \
        }                                                 \
    } while (0)

// Device pointer of an (optional) aclTensor; nullptr-safe.
static const void *SasDbgPtr(const aclTensor *t)
{
    return (t == nullptr) ? nullptr : t->devicePtr;
}

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor* cuSeqLensQOptional,
                               const aclTensor* cuSeqLensOriKvOptional,
                               const aclTensor* cuSeqLensCmpKvOptional,
                               const aclTensor* sequsedQOptional,
                               const aclTensor* sequsedKvOptional,
                               int64_t numHeadsQ,
                               int64_t numHeadsKv,
                               int64_t headDim,
                               int64_t batchSizeOptional,
                               int64_t maxSeqlenQOptional,
                               int64_t maxSeqlenKvOptional,
                               int64_t oriTopKOptional,
                               int64_t cmpTopKOptional,
                               int64_t cmpRatioOptional,
                               int64_t oriMaskModeOptional,
                               int64_t cmpMaskModeOptional,
                               int64_t oriWinLeftOptional,
                               int64_t oriWinRightOptional,
                               char *layoutQOptional,
                               char *layoutKvOptional,
                               bool hasOriKvOptional,
                               bool hasCmpKvOptional,
                               const aclTensor* metaData) {
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional,
    const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional,
    const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKvOptional,
    int64_t oriTopKOptional,
    int64_t cmpTopKOptional,
    int64_t cmpRatioOptional,
    int64_t oriMaskModeOptional,
    int64_t cmpMaskModeOptional,
    int64_t oriWinLeftOptional,
    int64_t oriWinRightOptional,
    char *layoutQOptional,
    char *layoutKvOptional,
    bool hasOriKvOptional,
    bool hasCmpKvOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  SAS_DBG("GWS enter: bs=" << batchSizeOptional << " maxSq=" << maxSeqlenQOptional
         << " maxSkv=" << maxSeqlenKvOptional << " headsQ=" << numHeadsQ
         << " headsKv=" << numHeadsKv << " headDim=" << headDim
         << " winL=" << oriWinLeftOptional << " winR=" << oriWinRightOptional
         << " topk=" << cmpTopKOptional << " ratio=" << cmpRatioOptional
         << " | ptrs: cuQ=" << SasDbgPtr(cuSeqLensQOptional)
         << " cuOriKv=" << SasDbgPtr(cuSeqLensOriKvOptional)
         << " cuCmpKv=" << SasDbgPtr(cuSeqLensCmpKvOptional)
         << " seqQ=" << SasDbgPtr(sequsedQOptional)
         << " seqKv=" << SasDbgPtr(sequsedKvOptional)
         << " OUT_metaData=" << SasDbgPtr(metaData));

  L2_DFX_PHASE_1(aclnnSparseAttnSharedkvMetadata,
                 DFX_IN(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional, sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSizeOptional,
                        maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, cmpRatioOptional, oriMaskModeOptional,
                        cmpMaskModeOptional, oriWinLeftOptional, oriWinRightOptional, layoutQOptional, layoutKvOptional,
                        hasOriKvOptional, hasCmpKvOptional),
                 DFX_OUT(metaData));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  SAS_DBG("GWS executor created");

  auto ret = ParamsCheck(
      cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional, sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSizeOptional,
      maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, cmpRatioOptional, oriMaskModeOptional,
      cmpMaskModeOptional, oriWinLeftOptional, oriWinRightOptional, layoutQOptional, layoutKvOptional,
      hasOriKvOptional, hasCmpKvOptional, metaData);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  SAS_DBG("GWS ParamsCheck ok");

  const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
  uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
  uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
  const char *socVersion = npuInfo.GetSocLongVersion().c_str();
  SAS_DBG("GWS platform: aicCoreNum=" << aicCoreNum << " aivCoreNum=" << aivCoreNum
         << " soc=" << (socVersion != nullptr ? socVersion : "<null>"));

  auto cuSeqLensQOptionalContiguous = l0op::Contiguous(cuSeqLensQOptional, uniqueExecutor.get());
  CHECK_RET(cuSeqLensQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  SAS_DBG("GWS contiguous cuQ: in=" << SasDbgPtr(cuSeqLensQOptional)
         << " out=" << SasDbgPtr(cuSeqLensQOptionalContiguous));
  auto cuSeqLensOriKvOptionalContiguous = l0op::Contiguous(cuSeqLensOriKvOptional, uniqueExecutor.get());
  CHECK_RET(cuSeqLensOriKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  SAS_DBG("GWS contiguous cuOriKv: in=" << SasDbgPtr(cuSeqLensOriKvOptional)
         << " out=" << SasDbgPtr(cuSeqLensOriKvOptionalContiguous));
  auto cuSeqLensCmpKvOptionalContiguous = l0op::Contiguous(cuSeqLensCmpKvOptional, uniqueExecutor.get());
  CHECK_RET(cuSeqLensCmpKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  SAS_DBG("GWS contiguous cuCmpKv: in=" << SasDbgPtr(cuSeqLensCmpKvOptional)
         << " out=" << SasDbgPtr(cuSeqLensCmpKvOptionalContiguous));
  auto sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
  CHECK_RET(sequsedQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  SAS_DBG("GWS contiguous seqQ: in=" << SasDbgPtr(sequsedQOptional)
         << " out=" << SasDbgPtr(sequsedQOptionalContiguous));
  auto sequsedKvOptionalContiguous = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
  CHECK_RET(sequsedKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  SAS_DBG("GWS contiguous seqKv: in=" << SasDbgPtr(sequsedKvOptional)
         << " out=" << SasDbgPtr(sequsedKvOptionalContiguous));

  auto output = l0op::SparseAttnSharedkvMetadata(
      cuSeqLensQOptionalContiguous, cuSeqLensOriKvOptionalContiguous, cuSeqLensCmpKvOptionalContiguous,
      sequsedQOptionalContiguous, sequsedKvOptionalContiguous, numHeadsQ, numHeadsKv, headDim, batchSizeOptional,
      maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, cmpRatioOptional, oriMaskModeOptional,
      cmpMaskModeOptional, oriWinLeftOptional, oriWinRightOptional, layoutQOptional, layoutKvOptional,
      hasOriKvOptional, hasCmpKvOptional, socVersion, aicCoreNum, aivCoreNum, metaData,
      uniqueExecutor.get());
  CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
  SAS_DBG("GWS l0op::SparseAttnSharedkvMetadata serialized ok, out=" << SasDbgPtr(output));

  *workspaceSize = 0;
  uniqueExecutor.ReleaseTo(executor);
  SAS_DBG("GWS ok: workspaceSize=0 executor=" << (void *)(*executor));
  return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnSparseAttnSharedkvMetadata(void *workspace, uint64_t workspaceSize,
                                aclOpExecutor *executor, aclrtStream stream) {
  SAS_DBG("RUN enter: executor=" << (void *)executor << " workspace=" << workspace
         << " wsSize=" << workspaceSize << " stream=" << (void *)stream);
  L2_DFX_PHASE_2(aclnnSparseAttnSharedkvMetadata);
  aclnnStatus runRet = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
  SAS_DBG("RUN end: ret=" << static_cast<int>(runRet) << " (0=success)");
  return runRet;
}

#ifdef __cplusplus
}
#endif
