#!/bin/bash

# Qwen2.5-7B Many-Shot KV Cache 运行脚本
# 使用 KV cache 复用机制进行 GSM8K 评估

echo "=========================================="
echo "Qwen2.5-7B Many-Shot KV Cache 实验"
echo "=========================================="
echo ""

# 激活 conda 环境
echo "激活 conda 环境: llm"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm

if [ $? -ne 0 ]; then
    echo "✗ 无法激活 conda 环境 'llm'"
    echo "请确保环境存在: conda env list"
    exit 1
fi

echo "✓ conda 环境 'llm' 已激活"
echo ""

# 设置工作目录
cd /data/oujie/oujie-data/shareShot/AdaCache

# ========== 参数配置 ==========
# 模型路径
MODEL_PATH="/data/oujie/models/Qwen/Qwen2.5-7B"

# 数据集配置
DATASET="AI-ModelScope/CoT-Collection:default"
# openai/gsm8k:main
TASK="iirc"
# task 种类还有 openbookqa 
# 实验参数
EVAL_SAMPLES=100        # 评测样本数
GLOBAL_POOL_SIZE=1024     # 全局示例池大小
WINDOW_SIZE=4             # 探针窗口大小
ENTROPY_THRESHOLD=1.0     # 熵阈值
MAX_PROBE_ROUNDS=256      # 最大探针轮数
GEN_TOKENS=4096           # 生成 token 数量
SEED=7678                   # 随机种子



# 运行配置
RUN_ID="qwen2.5_7B_${TASK}"
# 🔥 修复：OUTPUT_DIR 不能包含冒号等非法字符
OUTPUT_DIR="./Qwen2.5-7B"
MODE="cot"                # 模式: cot, io, paper

# ========== 开始实验 ==========
echo "实验配置:"
echo "  模型: ${MODEL_PATH}"
echo "  数据集: ${DATASET}"
echo "  评测样本数: ${EVAL_SAMPLES}"
echo "  全局池大小: ${GLOBAL_POOL_SIZE}"
echo "  窗口大小: ${WINDOW_SIZE}"
echo "  熵阈值: ${ENTROPY_THRESHOLD}"
echo "  模式: ${MODE}"
echo ""

ASCEND_RT_VISIBLE_DEVICES=0,1 python run_manyshot_kv.py \
    --mode ${MODE} \
    --models ${MODEL_PATH} \
    --datasets ${DATASET} \
    --tasks ${TASK} \
    --eval_samples ${EVAL_SAMPLES} \
    --global_pool_size ${GLOBAL_POOL_SIZE} \
    --window_size ${WINDOW_SIZE} \
    --entropy_threshold ${ENTROPY_THRESHOLD} \
    --max_probe_rounds ${MAX_PROBE_ROUNDS} \
    --gen_tokens ${GEN_TOKENS} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --run_id ${RUN_ID} \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 实验完成！"
    echo "=========================================="
    echo ""
    echo "查看结果："
    echo "  - 详细结果: ${OUTPUT_DIR}/manyshot_kv_*/latest/*.jsonl"
    echo "  - 指标汇总: ${OUTPUT_DIR}/manyshot_kv_*/latest/*_metrics.json"
    echo "  - 探针详情: ${OUTPUT_DIR}/manyshot_kv_*/latest/*_probe_details.jsonl"
    echo "  - 汇总表: ${OUTPUT_DIR}/summary_${RUN_ID}.csv"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ 实验失败"
    echo "=========================================="
    echo ""
    echo "请检查错误日志: ${OUTPUT_DIR}/logs/"
    exit 1
fi
