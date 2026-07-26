#!/bin/bash

ROLE="prefill"              # prefill / decode
HARDWARE_SERIES="A3"        # A2 (800I/800T A2) or A3 (800I/800T A3)
LOCAL_IP="80.5.17.37"
#NIC_NAME="enp194s0f0"
NIC_NAME="enx9c69d302197d"

#export VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE=60416
#export VLLM_ASCEND_ZBAL_BOOTSTRAP_URL="tcp://80.5.17.37:16989"
#export VLLM_ASCEND_ZBAL_MOE_ENABLE=1
export VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY=0
export VLLM_ASCEND_ZBAL_MOE_NVL_BYTES=10240
export VLLM_ASCEND_ZBAL_MOE_RDMA_BYTES=10240
export VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY_NUM_MAX_TOKENS_PER_RANK=1024

#MODEL_PATH="/home/weights/Qwen3-32B-W8A8/"
MODEL_PATH="/data/deepseekv4-flash-w8a8-mtp/"

SERVED_MODEL_NAME="dsv4"

#export ASCEND_RT_VISIBLE_DEVICES=4,5
#export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
#export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,
#export ZBAL_HCCL_OP="alltoall"
export ZBAL_HCCL_OP="reduce_scatter,alltoall,allgather"

export ASCEND_LAUNCH_BLOCKING=0
export MMC_LOCAL_CONFIG_PATH=/home/p00801009/vllm-ascend/vllm_test/mmc-local.conf

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

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
export HCCL_BUFFSIZE=2048
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
unset PYTORCH_NPU_ALLOC_CONF
export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0


# --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
# --no-disable-hybrid-kv-cache-manager \
# --async-scheduling \
NEW_ARGS=(
    --host 0.0.0.0
    --port 40060 \
    --model "$MODEL_PATH" \
    --max_model_len 104857 \
    --max-num-batched-tokens 8192 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.89 \
    --block-size 128 \
    --max-num-seqs 16 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
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
    --no-disable-hybrid-kv-cache-manager \
    --async-scheduling \
    --kv-transfer-config "$KV_CONFIG" \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 16,
                        "tp_size": 1
                }
        }
    }' \

    --additional-config '
    {
        "enable_cpu_binding": true,
        "enable_dsa_cp": true,
        "enable_shared_expert_dp": true
    }'
)
TS=$(date +"%Y%m%d_%H%M%S")
python -m vllm.entrypoints.openai.api_server "${NEW_ARGS[@]}" 2>&1 | tee log_${TS}_${ROLE}.log

echo "vLLM started. Log file: log_${ROLE}.log"
