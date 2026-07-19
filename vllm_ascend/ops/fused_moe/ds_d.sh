#!/bin/bash

ROLE="decode"              # prefill / decode
HARDWARE_SERIES="A3"        # A2 (800I/800T A2) or A3 (800I/800T A3)
LOCAL_IP="80.5.17.37"
NIC_NAME="enp194s0f0"

export VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE=60416
#export VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE=58368
export VLLM_ASCEND_ZBAL_BOOTSTRAP_URL="tcp://80.5.17.37:16999"
export VLLM_ASCEND_ZBAL_MOE_ENABLE=1
export VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY=1
export VLLM_ASCEND_ZBAL_MOE_NVL_BYTES=10240
export VLLM_ASCEND_ZBAL_MOE_RDMA_BYTES=10240
export VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY_NUM_MAX_TOKENS_PER_RANK=512

#MODEL_PATH="/home/weights/Qwen3-32B-W8A8/"
MODEL_PATH="/data/deepseekv4-flash-w8a8-mtp/"

SERVED_MODEL_NAME="dsv4"
P_DATA_PARALLEL_SIZE=4
P_TENSOR_PARALLEL_SIZE=4
D_DATA_PARALLEL_SIZE=16
D_TENSOR_PARALLEL_SIZE=1
#export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
#export ZBAL_HCCL_OP="alltoall"
export ASCEND_LAUNCH_BLOCKING=0

export MMC_LOCAL_CONFIG_PATH=/home/p00801009/vllm-ascend/vllm_test/mmc-local.conf
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
#export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

if [ "$ROLE" == "prefill" ]; then
    KV_ROLE="kv_producer"
    KV_PORT="30001"
    LOOKUP_RPC_PORT="0"
else
    KV_ROLE="kv_consumer"
    KV_PORT="30002"
    LOOKUP_RPC_PORT="1"
fi

echo "Starting vLLM on Series: $HARDWARE_SERIES, Role: $ROLE"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

if [ "$HARDWARE_SERIES" == "A2" ]; then
    echo 200000 > /proc/sys/vm/nr_hugepages
    export HCCL_IF_IP=$LOCAL_IP
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export TP_SOCKET_IFNAME=$NIC_NAME
    export HCCL_SOCKET_IFNAME=$NIC_NAME

elif [ "$HARDWARE_SERIES" == "A3" ]; then
    export ACL_OP_INIT_MODE=1
else
    echo "Error: Invalid HARDWARE_SERIES. Set to 'A2' or 'A3'."
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset PYTORCH_NPU_ALLOC_CONF
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1

KV_CONFIG='{
  "kv_connector": "MultiConnector",
  "kv_role": "'$KV_ROLE'",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "'$KV_ROLE'",
        "kv_port": "'$KV_PORT'",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": '$P_DATA_PARALLEL_SIZE',
            "tp_size": '$P_TENSOR_PARALLEL_SIZE'
          },
          "decode": {
            "dp_size": '$D_DATA_PARALLEL_SIZE',
            "tp_size": '$D_TENSOR_PARALLEL_SIZE'
          }
        }
      }
    ]
  }
}'

# --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
# --async-scheduling \
NEW_ARGS=(
    --port 40051
    --model "$MODEL_PATH" \
    --max_model_len 20480 \
    --max-num-batched-tokens 120 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.90 \
    --block-size 128 \
    --max-num-seqs 8 \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy 'prefetch' \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --no-disable-hybrid-kv-cache-manager \
    --async-scheduling \
    --kv-transfer-config "$KV_CONFIG" \
    --additional-config '
    {
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": true,
        "recompute_scheduler_enable":true
    }'
)
TS=$(date +"%Y%m%d_%H%M%S")
python -m vllm.entrypoints.openai.api_server "${NEW_ARGS[@]}" 2>&1 | tee log_${TS}_${ROLE}.log

echo "vLLM started. Log file: log_${ROLE}.log"
