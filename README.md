# PanguV1.0

本项目提供两类功能模块：
- AdaCache：基于 KV Cache 的多示例检索与复用（开发中）
- shot_baseline：多 shot 基线评测（已适配 Pangu）
- util：通用数据与指标模块（dataset_handlers、metrics 等）

## 目录结构
- `AdaCache/`：KV 池、探针、KV 拼装与评估入口（开发中）
- `shot_baseline/`：多 shot 评测入口与核心逻辑
  - `openPangu-Embedded-1B-V1.1/`：示例模型目录下的实验脚本与输出
  - `run_shot.py`、`shot_core.py`、`config.py`
- `util/`：通用模块（`dataset_handlers.py`、`metrics_utils.py`、`new_metrics.py`）

## 模块职责与工作流程
- `shot_baseline/`：面向“多示例提示（few/ many-shot）”的基线评测
  - 核心职责：
    - 以 CoT（Chain-of-Thought）、IO（输入-输出）、Paper（三段式示例）三种模式构造标准化 prompt
    - 加载本地模型（支持 NPU/CUDA/CPU），执行生成并记录每个样本的延迟、输入/输出 token 数、tokens/s、EM、数值准确率等指标
    - 输出 JSONL（逐样本）、metrics.json（汇总）、prompt_example（提示示例）、report（人类可读报告）、summary（CSV/MD 汇总）
  - 工作流程（简述）：
    1) 解析命令行参数与运行配置（`config.py`）
    2) 加载模型与分词器（强制本地加载，Pangu 模型启用 `trust_remote_code`）
    3) 从 `util/dataset_handlers.py` 获取数据集处理器，构造对应模式的 prompt
    4) 生成与评估，写出所有输出文件与汇总表

- `AdaCache/`：面向“KV Cache 复用”的多示例检索与自适应选择（开发中）
  - 核心职责：
    - 构建全局 KV 池并长期复用（prefill 阶段离线缓存）
    - 针对待测问题编码 Query 表征（平均 Q 向量），进行 Shot 排序与自适应探针（按窗口扩展、基于熵阈值停止）
    - 将所选示例的 K/V 与当前 prompt+query 的 K/V 在层维度正确拼接，进行最终生成
    - 保持本地模型加载与设备支持一致（NPU/CUDA/CPU），输出结构与日志与 shot_baseline 类似（更偏工程化）
  - 工作流程（简述）：
    1) 构建或载入 KV 池（`kv_pool_manager.py`）
    2) 获取 Query 表征并排序（`query_encoder.py`、`shot_ranker.py`）
    3) 探针选择（`probe_selector.py`）：按窗口逐轮扩展，基于输出分布熵达阈值即停止
    4) KV 拼装并生成（`kv_assembler.py`、`manyshot_kv_core.py`）
    5) 输出评测结果与汇总（`run_adacache.py`、`run_manyshot_kv.py`）
  - 架构与细节可参考 `MANYSHOT_KV_ARCHITECTURE.md`（来自原 AdaCache 代码库）

## 模式详解（CoT / IO / Paper）
- CoT（Chain-of-Thought）
  - 目标：通过示例中的“问题+逐步推理过程+最终答案”引导模型在作答时展示清晰的中间推理。
  - 示例结构：
    - 示例：
      - `Problem: <示例问题>`
      - `Solution: <逐步推理… 最后一行为 #### <答案>>`
    - 最终问题：
      - `Problem: <测试问题>`
      - `Solution:`（模型在此继续写过程并以 `#### <answer>` 收尾）
  - 适用：数学推理、需要过程解释的任务。

- IO（Input-Output）
  - 目标：用最简的“输入-输出”示例教模型映射关系，不暴露推理过程。
  - 示例结构：
    - 示例：
      - `Problem: <示例问题>`
      - `Answer: <答案>`
    - 最终问题：
      - `Problem: <测试问题>`
      - `Answer:`（模型直接给出结果）
  - 适用：只需最终答案、希望减少冗长输出的场景。

- Paper（三段式提示）
  - 目标：分阶段规范输出格式，提升格式遵守与稳健性。
  - 示例结构：
    - 引导 + 只列题目（question-only shots）：
      - `You will be provided Problems similar to the ones below:`
      - 多个 `Problem: <题目>`
      - 分隔线 `—`
    - 输出规范 + 完整示例（fullshots）：
      - `Now, I am going to give you a series of demonstrations...`
      - `When you respond, think step by step, but your last line must be exactly of the form '#### <final_answer>'.`
      - 每个完整示例包含：`Problem:` 与 `Solution:`（结尾严格 `#### <final_answer>`）
      - 分隔线 `—`
    - 最终问题：
      - `Problem: <测试问题>`
      - `Solution:`
  - 适用：强调输出格式一致性、需同时利用弱提示（只列题目）与强提示（完整示例）。

## 我们的工作核心
- 标准化多模式提示构造（CoT/IO/Paper），结合统一的数据处理与答案抽取逻辑，确保评测可靠性与可复现性。
- 本地模型加载与设备自适配：支持 NPU/CUDA/CPU，Pangu 模型强制信任远程代码实现并本地读取权重。
- 评测与可观测性：记录延迟与 tokens/s 等性能指标，同时输出逐样本与汇总结果，便于横向比较不同 `num_shots` 与模式。
- 面向 KV 复用的工程化架构（AdaCache，开发中）：以 KV 池 + 探针选择 + KV 拼装实现多示例复用与性能-效果折中。


## 环境配置
- 推荐使用 conda 环境
- 支持设备：NPU（Ascend 910/910B）、CUDA（NVIDIA GPU）、CPU
- 本地模型加载：代码默认使用本地模型并启用 `trust_remote_code=True` 与 `local_files_only=True`
- 可选的 HuggingFace 镜像与本地缓存（按需配置）：
  - `HF_ENDPOINT=https://hf-mirror.com`
  - `HF_HOME=<your_local_hf_home>`
  - `HF_DATASETS_CACHE=<your_local_datasets_cache>`

### NPU（Ascend）
- 安装 Ascend Toolkit 与匹配版本的 `torch-npu`
- 使用本地模型目录，避免在线拉取

### CUDA（NVIDIA）
- 安装 CUDA 对应的 PyTorch 版本（`torch`）
- 其余 Python 依赖参考 `environment.yml`

### CPU
- 安装 CPU 版 PyTorch（`torch`）与其余依赖

## 模型准备
- 将 Pangu 模型本地化，例如：`models/pangu/FreedomIntelligence/openPangu-Embedded-1B-V1.1`
- 运行时通过 `--models` 或 `--model_root` 指定本地模型路径

## Shot Baseline 使用
- 位置：`shot_baseline/openPangu-Embedded-1B-V1.1`
- Shell 脚本：`run_gsm8k.sh`
  - 示例：
    - `cd shot_baseline/openPangu-Embedded-1B-V1.1`
    - `bash run_gsm8k.sh`
- Python 入口：`shot_baseline/run_shot.py`
  - 关键参数：
    - `--models` 模型路径（多个用逗号分隔）
    - `--datasets` 数据集与子集 `dataset:subset`（例如 `openai/gsm8k:main`）
    - `--mode` 评测模式：`cot`、`io`、`paper`
    - `--num_shots` CoT/IO 模式的示例数
    - `--paper_k_full`、`--paper_num_questions` Paper 模式配置
    - `--eval_samples` 评测样本数
    - `--gen_tokens` 生成的最大 token 数
    - `--seed` 随机种子
    - `--model_root` 模型根目录（默认 `models`）
    - `--output_dir` 输出目录（默认 `./outputs`）
    - `--run_id` 运行标识
    - `--device` 设备类型：`npu`、`cuda`、`cpu`（不填自动检测）
    - `--use_sharding` NPU 多卡分片（默认启用）
  - 示例命令：
    - `python shot_baseline/run_shot.py --models models/pangu/FreedomIntelligence/openPangu-Embedded-1B-V1.1 --datasets openai/gsm8k:main --mode cot --num_shots 8 --eval_samples 100 --gen_tokens 512 --seed 42 --model_root models --output_dir outputs --run_id pangu_shot_test --verbose`

### 输出说明
- 输出目录：`shot_baseline/openPangu-Embedded-1B-V1.1/outputs`（脚本会自动创建）
- 典型文件：
  - `*.jsonl`：逐样本记录（问题、预测、指标、耗时与 token 数）
  - `*_metrics.json`：汇总指标（准确率、误差、平均延迟等）
  - `*_pool_info.json`：示例池信息预览
  - `*_prompt_example.txt`：完整 Prompt 示例
  - `*_report.txt`：人类可读的评测报告
  - `summary_*.csv`、`summary_*.md`：汇总表

## AdaCache 概览（开发中）
- 位置：`AdaCache/`
- 入口脚本：
  - `run_adacache.py`：自适应示例选择评测入口
  - `run_manyshot_kv.py`：Many-Shot KV Cache 评测入口
- 架构参考：`MANYSHOT_KV_ARCHITECTURE.md`（来自原 AdaCache 代码库）
- 简要示例：
  - `python AdaCache/run_manyshot_kv.py --mode cot --global_pool_size 1024 --window_size 4 --entropy_threshold 4.5 --eval_samples 50 --models models/pangu/FreedomIntelligence/openPangu-Embedded-1B-V1.1 --datasets openai/gsm8k:main --output_dir outputs --run_id pangu_kv_test`

## Quickstart
1. 创建并激活 conda 环境：
   - `conda env create -f environment.yml`
   - `conda activate pangu`
2. 准备本地模型：
   - 下载并放置 Pangu 模型，例如：`models/pangu/FreedomIntelligence/openPangu-Embedded-1B-V1.1`
3. 运行 Shot Baseline：
   - `cd shot_baseline/openPangu-Embedded-1B-V1.1`
   - `bash run_gsm8k.sh`
4. 查看输出：
   - `outputs/` 下的 `jsonl`、`metrics.json`、`report.txt`、汇总 `summary_*.csv/md`

## 常见问题
- `ModuleNotFoundError: util`
  - 以项目根作为工作路径运行；脚本已自动将项目根加入 `sys.path`
- NPU 报错（Ascend Toolkit）
  - 检查 Toolkit 安装与权限、环境变量（`PATH`、`LD_LIBRARY_PATH`）；确保 `torch-npu` 版本匹配
- 本地模型
  - 确保本地路径存在；代码使用 `local_files_only=True` 与 `trust_remote_code=True`
- 长文本或重复输出
  - 模型停止符依赖具体模型；已在评估逻辑中控制生成长度与输出格式，必要时调整 `gen_tokens`
