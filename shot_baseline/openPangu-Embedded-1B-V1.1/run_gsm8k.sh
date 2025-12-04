
echo "========================================="
echo "Shot Baseline - 多 shot 数测试"
echo "========================================="

# 使用的数据集
DATASET="openai/gsm8k:main"
MODEL="/data/oujie/models/pangu/FreedomIntelligence/openPangu-Embedded-1B-V1.1"
EVAL_SAMPLES=10
RUN_ID="penPangu-Embedded-1B_gsm8k"

# ============================================
# CoT 模式：测试 2, 4, 8,16,32,64，128,256,512,1024shot
# ============================================
echo ""
echo "========================================="
echo "CoT 模式：测试不同 shot 数"
echo "========================================="

for NUM_SHOTS in 8 16 32 ; do
    echo ""
    echo "--- CoT ${NUM_SHOTS}-shot ---"
    ASCEND_RT_VISIBLE_DEVICES=2,3 python /data/oujie/PanguV1.0/shot_baseline/run_shot.py \
        --models $MODEL \
        --datasets $DATASET \
        --mode cot \
        --num_shots $NUM_SHOTS \
        --eval_samples $EVAL_SAMPLES \
        --gen_tokens 512 \
        --seed 7672 \
        --run_id ${RUN_ID}_cot_${NUM_SHOTS}shot \
        --verbose
done

# # ============================================
# # IO 模式：测试 2, 4, 8,16,32,64，128,256,512,1024shot
# # ============================================
# echo ""
# echo "========================================="
# echo "IO 模式：测试不同 shot 数"
# echo "========================================="

# for NUM_SHOTS in 2 4 8 16 32 64 128 256 512 1024; do
#     echo ""
#     echo "--- IO ${NUM_SHOTS}-shot ---"
#     ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 python /data/oujie/PanguV1.0/shot_baseline/run_shot.py \
#         --models $MODEL \
#         --datasets $DATASET \
#         --mode io \
#         --tasks $TASK \
#         --num_shots $NUM_SHOTS \
#         --eval_samples $EVAL_SAMPLES \
#         --gen_tokens 1024 \
#         --seed 7672 \
#         --run_id ${RUN_ID}_io_${NUM_SHOTS}shot \
#         --verbose
# done

# # ============================================
# # Paper 模式：测试不同配置
# # ============================================
# echo ""
# echo "========================================="
# echo "Paper 模式：测试不同配置"
# echo "========================================="

# # 定义要测试的配置
# # 格式：fullshot_count:question_count
# paper_configs=(
#     "4:2"
#     "4:4"
#     "4:8"
#     "4:16"
#     "4:32"
#     "4:64"
#     "4:128"
#     "4:256"
#     "4:512"
#     "4:1024"
# )

# # 循环执行不同的 Paper 模式配置
# for config in "${paper_configs[@]}"; do
#     # 解析配置
#     IFS=':' read -r fullshot questions <<< "$config"
    
#     echo ""
#     echo "--- Paper ${fullshot}+${questions} ---"
#     ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 python /data/oujie/PanguV1.0/shot_baseline/run_shot.py \
#         --models $MODEL \
#         --datasets $DATASET \
#         --mode paper \
#         --paper_k_full $fullshot \
#         --paper_num_questions $questions \
#         --tasks $TASK \
#         --eval_samples $EVAL_SAMPLES \
#         --gen_tokens 1024 \
#         --seed 7672 \
#         --run_id ${RUN_ID}_paper_${fullshot}_${questions} \
#         --verbose
# done

# echo ""
# echo "========================================="
# echo "✓ 所有测试完成！"
# echo "========================================="
