#!/usr/bin/env zsh
set -euo pipefail

# ==============================
# Flint INT4 - vLLM 4x Tesla T4
# ==============================

# 修改为你的模型实际路径
MODEL_PATH="/home/lhc/models/flint-gguf/flint-UD-Q4_K_XL.gguf"
CONFIG_PATH="/home/lhc/models/flint-gguf"

# 服务配置
HOST="0.0.0.0"
PORT=8000
# GGUF K-quant blocks require 256-element expert input partitions.
TP_SIZE=2
PP_SIZE=2

# T4 显存配置
GPU_MEMORY_UTIL=0.85
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# 使用 4 张 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 减少 tokenizer 并行警告
export TOKENIZERS_PARALLELISM=false

# NCCL 调试配置
export NCCL_DEBUG=WARN

# 检查模型文件
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查 vLLM
VLLM_BIN="${0:A:h}/.venv/bin/vllm"
if [[ ! -x "$VLLM_BIN" ]]; then
    echo "未找到 vllm，请先激活对应 Python 环境"
    exit 1
fi

echo "=================================="
echo "Starting Flint INT4 with vLLM"
echo "Model: $MODEL_PATH"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Tensor Parallel: $TP_SIZE"
echo "Pipeline Parallel: $PP_SIZE"
echo "Port: $PORT"
echo "=================================="

exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --hf-config-path "$CONFIG_PATH" \
    --tokenizer "$CONFIG_PATH" \
    --trust-remote-code \
    --served-model-name flint-int4 \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    --dtype float16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enforce-eager
