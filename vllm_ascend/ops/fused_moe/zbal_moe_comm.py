#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""ZBAL MoE communication method.

This module implements the MoE communication method using ZBAL Buffer's
dispatch and combine interfaces, providing DeepEP-like functionality
for high-throughput intranode all-to-all communication on Ascend NPUs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from typing import Optional

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm_ascend.quantization.quant_type import QuantType
import torch.distributed as dist

import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.zbal_utils import is_zbal_enabled
from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEPrepareOutput,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatcher
from vllm_ascend.ops.fused_moe.zbal_moe_adapter import ZBALMoEAdapter


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MoEZBALCombineMetadata:
    """Combine metadata for ZBAL MoE communication.

    Carries the information needed by :meth:`TokenDispatcherWithZBAL.token_combine`
    to reverse the dispatch operation.

    Attributes:
        is_low_latency: Whether the corresponding dispatch used the
            low-latency path. Combine must use the matching path to avoid
            handle/buffer mismatches that would trigger MTE errors.
    """

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    handle: tuple
    # In low-latency path this holds a device tensor (kept on-device so the
    # dispatch/combine pipeline stays cudagraph-capturable). In normal path
    # it may be a Python list returned by the C++ buffer.dispatch.
    num_recv_tokens_per_expert_list: list | torch.Tensor
    is_low_latency: bool = False


class TokenDispatcherWithZBAL(MoETokenDispatcher[MoEZBALCombineMetadata]):
    """Token dispatcher using ZBAL Buffer for dispatch/combine.

    This dispatcher uses the ZBAL Buffer's high-throughput intranode
    all-to-all communication to dispatch tokens to expert ranks and
    combine the results back.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device_group = get_ep_group().device_group
        self.num_experts = kwargs.get("num_experts", 0)
        # hidden_size is required for ZBAL buffer initialization.
        self.hidden_size = kwargs.get("hidden_size", 0)
        if self.hidden_size == 0:
            raise ValueError(
                "hidden_size must be provided to TokenDispatcherWithZBAL"
            )

        # Read buffer sizes from environment variables.
        self.low_latency_mode = envs_ascend.VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY
        # Static cap for low-latency dispatch buffer. Read once at init and
        # shared with the adapter. Batches larger than this cap transparently
        # fall back to the normal dispatch/combine path.
        self.low_latency_num_max_tokens_per_rank = (
            envs_ascend.VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY_NUM_MAX_TOKENS_PER_RANK
        )

        self._adapter = None
        # When a normal dispatch runs on a buffer that was previously in
        # low-latency mode, the low-latency buffer becomes dirty (its
        # zero-initialized region is overwritten). We must call
        # `clean_low_latency_buffer` before the next low-latency dispatch.
        # This flag tracks that need so we can clean lazily on the next
        # low-latency forward.
        self._needs_clean_before_low_latency = False

        self.ep_rank_id = get_ep_group().rank_in_group
        self.ep_world_size = get_ep_group().world_size

    def ensure_adapter_constructed(self) -> None:
        """Explicitly construct the ZBALMoEAdapter (and its C++ Buffer) if it
        has not been constructed yet.

        This is intended to be called AFTER ``zbal_init`` (which carves the
        GVA heap) and BEFORE ACL graph capture begins. Calling it eagerly at
        that point guarantees:

        1. The Buffer's RDMA memory is allocated from the correct GVA pool
           (not from DMA VMM, which would happen if construction occurred
           before ``zbal_init``).
        2. The first ACL graph capture does not implicitly trigger lazy
           construction inside ``torch.npu.graph()``, which could capture
           allocator ops into the graph and corrupt replay.
        3. The low-latency buffer is pre-cleaned so the first captured
           forward sees a zero-initialized RDMA region.

        After this call, ``self._adapter`` is guaranteed to be non-None.
        Subsequent ``token_dispatch`` calls reuse the constructed adapter.
        """
        if self._adapter is not None:
            return
        self._adapter = ZBALMoEAdapter(
            group=self.device_group,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            low_latency_mode=self.low_latency_mode,
        )
        if self.low_latency_mode:
            self._adapter.clean_low_latency_buffer(
                self.low_latency_num_max_tokens_per_rank,
                self.hidden_size,
                self.num_experts,
            )
            self._needs_clean_before_low_latency = False
        logger.info(
            "[TokenDispatcherWithZBAL] Pre-constructed ZBALMoEAdapter "
            "(low_latency=%s, cap=%d) before graph capture.",
            self.low_latency_mode, self.low_latency_num_max_tokens_per_rank,
        )

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[MoEZBALCombineMetadata]:
        if self._adapter is None:
            self._adapter = ZBALMoEAdapter(
                group=self.device_group,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                low_latency_mode=self.low_latency_mode,
            )

        """Dispatch tokens to expert ranks using ZBAL Buffer."""
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids

        # ZBAL dispatch kernel routes tokens by dstExpertId / moeExpertNumPerRank,
        # which assumes uniform expert distribution. EPLB (dynamic expert
        # rebalancing) uses log2phy to remap expert indices, breaking this
        # assumption. expert_map alone (without log2phy) is fine — it just marks
        # which experts are local and ZBAL handles routing correctly.
        if token_dispatch_input.routing.log2phy is not None:
            raise NotImplementedError(
                "ZBAL MoE communication does not support EPLB (log2phy) yet. "
                "Please disable dynamic EPLB or use a different MoE comm method."
            )

        # ZBAL combine kernel always applies topk_weights (weighted reduction).
        # When apply_router_weight_on_input=True, weights are already
        # pre-multiplied into hidden_states, so combine must use ones to avoid
        # double weighting.
        apply_router_weight_on_input = (
            token_dispatch_input.routing.apply_router_weight_on_input
        )
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, (
                "`topk_weights` should be in shape (num_tokens, topk)"
            )
            _, topk = topk_weights.shape
            assert topk == 1, (
                "Only support topk=1 when `apply_router_weight_on_input` is True"
            )
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            combine_weights = torch.ones_like(topk_weights, dtype=torch.float32)
        else:
            # ZBAL C++ combine kernel requires float32 topk_weights.
            combine_weights = topk_weights.to(torch.float32)

        # ZBAL dispatch requires int64 topk_idx.
        topk_idx = topk_ids.to(torch.int64)

        # Decide whether to use the low-latency path for this forward.
        # The low-latency RDMA buffer is allocated ONCE at adapter init time
        # based on `low_latency_num_max_tokens_per_rank` (a static env-var
        # value). If the current batch exceeds this cap, we must NOT use the
        # low-latency path — otherwise the C++ dispatch kernel would write
        # past the pre-allocated buffer and trigger MTE address out-of-bounds
        # errors in the subsequent combine. Falling back to the normal path
        # (which uses dynamic layout computation) preserves correctness at
        # the cost of higher latency for that single forward.
        actual_tokens = hidden_states.shape[0]
        prefer_low_latency = self.low_latency_mode
        use_low_latency = (
            prefer_low_latency
            and actual_tokens <= self.low_latency_num_max_tokens_per_rank
        )
        if prefer_low_latency and not use_low_latency:
            # The normal (non-low-latency) dispatch path uses dynamic layout
            # computation and returns tensors whose shapes depend on the
            # actual per-expert token distribution. This is NOT capturable
            # by ACL graph: replay would reuse the captured shapes instead
            # of the new batch's shapes, producing wrong results or crashes.
            # During graph capture we must therefore raise instead of fall
            # back, so the user can fix the config (raise the cap or shrink
            # the capture batch size) rather than getting a silent corrupt
            # graph.
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(
                    f"[TokenDispatcherWithZBAL] Batch size {actual_tokens} "
                    f"exceeds low_latency cap "
                    f"{self.low_latency_num_max_tokens_per_rank} during "
                    f"ACL graph capture. The normal dispatch fallback is "
                    f"not graph-safe (dynamic shapes). Please either: "
                    f"(1) increase "
                    f"VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY_NUM_MAX_TOKENS_PER_RANK "
                    f"to >= {actual_tokens}, or (2) reduce "
                    f"cudagraph_capture_sizes / max_num_seqs so captured "
                    f"batch sizes fit within the cap."
                )
            logger.warning(
                "[TokenDispatcherWithZBAL] Batch %d exceeds low_latency cap "
                "%d; falling back to normal dispatch for this forward. "
                "To avoid frequent fallbacks, increase "
                "VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY_NUM_MAX_TOKENS_PER_RANK.",
                actual_tokens, self.low_latency_num_max_tokens_per_rank,
            )
            # Normal dispatch leaves the low-latency buffer dirty. Mark it
            # so the next low-latency forward knows to clean before use.
            self._needs_clean_before_low_latency = True

        # Use standard or low-latency dispatch based on the decision above.
        recv_x_scales = None
        if use_low_latency:
            # If the previous forward fell back to normal dispatch, the
            # low-latency buffer's zero-initialized region was overwritten.
            # ZBAL's low-latency kernels require that region to be zero, so
            # we must clean it before issuing the next low-latency dispatch.
            # See ZBAL Buffer.clean_low_latency_buffer docs.
            if self._needs_clean_before_low_latency:
                self._adapter.clean_low_latency_buffer(
                    self.low_latency_num_max_tokens_per_rank,
                    self.hidden_size,
                    self.num_experts,
                )
                self._needs_clean_before_low_latency = False

            # Always pass the static cap as `num_max_tokens_per_rank`, NEVER
            # `hidden_states.shape[0]`. The C++ runtime uses this value to
            # compute RDMA slot offsets; passing a dynamic value would
            # desynchronize buffer size from actual write addresses.
            num_max_tokens_per_rank = self.low_latency_num_max_tokens_per_rank
            recv_x, recv_count, handle_dict, event = self._adapter.low_latency_dispatch(
                x=hidden_states,
                topk_idx=topk_idx,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                use_fp8=False,
            )
            # low_latency_dispatch returns recv_count with shape
            # [num_local_experts], which is the per-expert token count
            # needed by npu_grouped_matmul as group_list.
            # Keep recv_count as a device tensor to avoid a D2H sync
            # (`.tolist()`) that would break ACL graph capture. The downstream
            # `npu_grouped_matmul` and `cumsum_group_list` accept device
            # tensors for group_list, and the low-latency combine path does
            # not consume `num_recv_tokens_per_expert_list` at all.
            group_list = recv_count.to(torch.int64)
            num_recv_tokens_per_expert_list = group_list
        else:
            # Pass topk_weights so zbal can forward them to receiving ranks.
            # In standard mode, the handle stores these weights for combine.
            recv_x, recv_topk_idx, handle_dict, recv_x_scales = self._adapter.dispatch(
                x=hidden_states,
                topk_idx=topk_idx,
                topk_weights=combine_weights,
            )
            # Build group_list for MLP computation.
            # npu_grouped_matmul requires int64 group_list.
            num_recv_tokens_per_expert_list = handle_dict.get(
                "num_recv_tokens_per_expert_list", []
            )
            if num_recv_tokens_per_expert_list:
                group_list = torch.tensor(
                    num_recv_tokens_per_expert_list,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
            else:
                group_list = torch.zeros(1, dtype=torch.int64, device=hidden_states.device)

        combine_metadata = MoEZBALCombineMetadata(
            topk_ids=topk_ids,
            topk_weights=combine_weights,
            handle=handle_dict["handle"],
            num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
            is_low_latency=use_low_latency,
        )

        logger.debug(
            "[TokenDispatcherWithZBAL] Dispatch completed: recv_x.shape=%s",
            recv_x.shape,
        )

        return MoETokenDispatchOutput(
            hidden_states=recv_x,
            group_list=group_list,
            group_list_type=1,
            combine_metadata=combine_metadata,
            dynamic_scale=recv_x_scales,
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: MoEZBALCombineMetadata,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine tokens from expert ranks using ZBAL Buffer."""
        logger.debug(
            "[TokenDispatcherWithZBAL] Combining tokens: hidden_states.shape=%s",
            hidden_states.shape,
        )

        topk_weights = combine_metadata.topk_weights
        # ZBAL C++ combine kernel requires float32 topk_weights.
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)
        handle = combine_metadata.handle

        # Use the combine path that matches the dispatch path recorded in
        # `combine_metadata.is_low_latency`. The two paths use different
        # C++ kernels with different handle layouts; mixing them would
        # dereference invalid buffer offsets.
        if combine_metadata.is_low_latency:
            topk_idx = combine_metadata.topk_ids.to(torch.int64)
            combined_x, event, hook = self._adapter.low_latency_combine(
                x=hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                handle_dict={"handle": handle},
            )
        else:
            combined_x = self._adapter.combine(
                x=hidden_states,
                handle_dict={"handle": handle},
                topk_weights=topk_weights,
            )

        logger.debug(
            "[TokenDispatcherWithZBAL] Combine completed: combined_x.shape=%s",
            combined_x.shape,
        )

        return combined_x


class PrepareAndFinalizeWithZBAL(PrepareAndFinalizeWithAll2All):
    """PrepareAndFinalize for ZBAL MoE communication.

    ZBAL's ProcessGroup all_gather does not support tensors with different
    sizes across TP ranks. This class pads num_tokens to a multiple of
    tp_size so that torch.tensor_split produces equal-sized slices.
    """

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        padded_hidden_states_shape = hidden_states.shape
        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            self.num_tokens, _ = hidden_states.shape
            pad_size = (self.tp_size - self.num_tokens % self.tp_size) % self.tp_size

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def pad_and_split_input_ids(self, input_ids):
        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            pad_size = (self.tp_size - self.num_tokens % self.tp_size) % self.tp_size
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            if self.tp_size > 1:
                input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
                input_ids = input_ids[self.tp_rank]
        return input_ids


class ZBALCommImpl(MoECommMethod):
    """MoE communication method using ZBAL Buffer.

    This implementation uses ZBAL's high-throughput intranode all-to-all
    communication for dispatch and combine operations, providing DeepEP-like
    functionality on Ascend NPUs.

    Requirements:
    - ZBAL must be enabled (VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0)
    - VLLM_ASCEND_ZBAL_MOE_ENABLE must be set to 1
    - Expert parallel must be enabled with EP size > 1
    """

    def __init__(self, moe_config: FusedMoEConfig):
        if not is_zbal_enabled():
            raise RuntimeError(
                "ZBAL is not enabled. Please set VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE > 0"
            )
        if not envs_ascend.VLLM_ASCEND_ZBAL_MOE_ENABLE:
            raise RuntimeError(
                "ZBAL MoE is not enabled. Please set VLLM_ASCEND_ZBAL_MOE_ENABLE=1"
            )

        # Resolve hidden_size BEFORE super().__init__(): the parent ctor calls
        # _get_token_dispatcher(), which reads self._hidden_size. Setting it
        # afterwards triggers AttributeError.
        self._hidden_size = self._resolve_hidden_size(moe_config)

        super().__init__(moe_config)

        logger.info(
            "[ZBALCommImpl] Initialized ZBAL MoE communication method "
            "(low_latency=%s, hidden_size=%s)",
            envs_ascend.VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY, self._hidden_size,
        )

    def _resolve_hidden_size(self, moe_config: FusedMoEConfig) -> int:
        """Resolve hidden_size from MoE config.

        ZBAL Buffer initialization requires the hidden dimension size.
        We derive it from the weight shapes stored in moe_config.
        """
        # Try common attributes that expose hidden size.
        for attr in ("hidden_size", "hidden_dim"):
            val = getattr(moe_config, attr, None)
            if val and isinstance(val, int) and val > 0:
                return val
        # Fallback: use a reasonable default if available from weights.
        logger.warning(
            "[ZBALCommImpl] Could not resolve hidden_size from moe_config, "
            "falling back to 0. ZBAL Buffer initialization may fail."
        )
        return 0

    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        return TokenDispatcherWithZBAL(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
            hidden_size=self._hidden_size,
        )

    def ensure_adapter_constructed(self) -> None:
        """Pre-construct the ZBAL Buffer before ACL graph capture.

        Delegates to :meth:`TokenDispatcherWithZBAL.ensure_adapter_constructed`.
        Safe to call multiple times; no-op after the first call.
        """
        self.token_dispatcher.ensure_adapter_constructed()

    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        # ZBAL's ProcessGroup requires all_gather tensors to have identical
        # sizes across TP ranks. PrepareAndFinalizeWithZBAL pads num_tokens
        # to a multiple of tp_size to guarantee uniform tensor_split slices.
        return PrepareAndFinalizeWithZBAL(self.moe_config)

    def pad_and_split_input_ids(self, input_indx):
        return self.prepare_finalize.pad_and_split_input_ids(input_indx)
