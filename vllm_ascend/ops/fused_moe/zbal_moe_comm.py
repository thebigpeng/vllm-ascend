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

import torch
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.zbal_utils import is_zbal_enabled
from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
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
    """

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    handle: tuple
    num_recv_tokens_per_expert_list: list


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
        self.num_nvl_bytes = envs_ascend.VLLM_ASCEND_ZBAL_MOE_NVL_BYTES
        self.num_rdma_bytes = envs_ascend.VLLM_ASCEND_ZBAL_MOE_RDMA_BYTES
        self.low_latency_mode = envs_ascend.VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY

        self._adapter = None

        self.ep_rank_id = get_ep_group().rank_in_group
        self.ep_world_size = get_ep_group().world_size

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[MoEZBALCombineMetadata]:
        if self._adapter is None:
            self._adapter = ZBALMoEAdapter(
                group=self.device_group,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                num_nvl_bytes=self.num_nvl_bytes,
                num_rdma_bytes=self.num_rdma_bytes,
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

        logger.debug(
            "[TokenDispatcherWithZBAL] Dispatching tokens: "
            "hidden_states.shape=%s, topk_ids.shape=%s",
            hidden_states.shape, topk_ids.shape,
        )

        # ZBAL dispatch requires int64 topk_idx.
        topk_idx = topk_ids.to(torch.int64)

        # Use standard or low-latency dispatch based on config.
        if envs_ascend.VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY:
            num_max_tokens_per_rank = hidden_states.shape[0]
            recv_x, recv_count, handle_dict, event = self._adapter.low_latency_dispatch(
                x=hidden_states,
                topk_idx=topk_idx,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                use_fp8=False,
            )
        else:
            # Pass topk_weights so zbal can forward them to receiving ranks.
            # In standard mode, the handle stores these weights for combine.
            recv_x, recv_topk_idx, handle_dict = self._adapter.dispatch(
                x=hidden_states,
                topk_idx=topk_idx,
                topk_weights=combine_weights,
            )

        # Build group_list for MLP computation.
        num_recv_tokens_per_expert_list = handle_dict.get(
            "num_recv_tokens_per_expert_list", []
        )
        if num_recv_tokens_per_expert_list:
            group_list = torch.tensor(
                num_recv_tokens_per_expert_list,
                dtype=torch.int32,
                device=hidden_states.device,
            )
        else:
            group_list = torch.zeros(1, dtype=torch.int32, device=hidden_states.device)

        combine_metadata = MoEZBALCombineMetadata(
            topk_ids=topk_ids,
            topk_weights=combine_weights,
            handle=handle_dict["handle"],
            num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
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

        # Use standard or low-latency combine based on config.
        if envs_ascend.VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY:
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

    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        # ZBAL reuses the All2All prepare/finalize logic for tensor
        # padding, slicing, and reduction across TP/DP groups.
        return PrepareAndFinalizeWithAll2All(self.moe_config)
