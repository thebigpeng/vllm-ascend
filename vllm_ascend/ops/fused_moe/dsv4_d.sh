#!/bin/bash

ROLE="decode"              # prefill / decode
HARDWARE_SERIES="A3"        # A2 (800I/800T A2) or A3 (800I/800T A3)
LOCAL_IP="80.5.17.36"
#NIC_NAME="enp194s0f0"
NIC_NAME="enp48s3u1u1"
export VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE=60416

#export VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE=53248
export VLLM_ASCEND_ZBAL_BOOTSTRAP_URL="tcp://80.5.17.36:16889"
export VLLM_ASCEND_ZBAL_MOE_ENABLE=1
export VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY=0
export VLLM_ASCEND_ZBAL_MOE_NVL_BYTES=10240
export VLLM_ASCEND_ZBAL_MOE_RDMA_BYTES=10240
#export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
#export ASCEND_RT_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"
#export ZBAL_HCCL_OP="alltoall,barrier,reduce_scatter,scatter,broadcast,allgather,allreduce,send,recv"
export ZBAL_HCCL_OP="reduce_scatter,alltoall,allgather"
#export ZBAL_PROF_ENABLE=1

#MODEL_PATH="/home/weights/Qwen3-32B-W8A8/"
MODEL_PATH="/home/weights/deepseekv4-flash-w8a8-mtp/"

SERVED_MODEL_NAME="dsv4"

export ASCEND_LAUNCH_BLOCKING=0
export MMC_LOCAL_CONFIG_PATH=/home/p00801009/vllm-ascend/vllm_test/mmc-local.conf

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_APPLY_DSV4_PATCH=1

if [ "$ROLE" == "prefill" ]; then
    KV_ROLE="kv_producer"
    KV_PORT="30011"
    LOOKUP_RPC_PORT="0"
else
    KV_ROLE="kv_consumer"
    KV_PORT="30012"
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


# --async-scheduling \
# --no-disable-hybrid-kv-cache-manager \
# --enforce-eager \
NEW_ARGS=(
    --port 40051
    --model "$MODEL_PATH" \
    --served-model-name dsv4 \
    --block-size 128 \
    --max-num-seqs 16 \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --max-model-len 10857 \
    --max-num-batched-tokens 60 \
    --no-disable-hybrid-kv-cache-manager \
    --async-scheduling \
    --no-enable-prefix-caching \
    --safetensors-load-strategy 'prefetch' \
    --trust-remote-code \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/p00801009/vllm-ascend/vllm_test/vllm_profile", "torch_profiler_with_stack": false}'
    --tokenizer-mode deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
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
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_cpu_binding":true,
        "multistream_overlap_shared_expert":true,
        "recompute_scheduler_enable":true
    }'
)

python -m vllm.entrypoints.openai.api_server "${NEW_ARGS[@]}" 2>&1 | tee log_${ROLE}.log

echo "vLLM started. Log file: log_${ROLE}.log"
