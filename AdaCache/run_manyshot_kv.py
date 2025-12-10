"""
Many-Shot KV Cache 主运行脚本
基于 KV cache 的完整检索与复用系统 (华为 910B NPU)

完整流程:
1. 离线 Prefilling: 构建 1024-shot KV cache 池
2. Query 编码: 提取平均 Q 向量
3. Shot 排序: Token级打分 -> Shot级聚合
4. 探针选择: 按窗口n逐轮扩展，熵判断停止
5. KV 拼装: shots KV + prompt+query KV
6. 最终生成: 使用拼装的 KV cache
"""
import sys
import os
import argparse
import logging
import torch
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "60")
# Merge: 修改 HF_HOME 路径为共享存储路径
Original: os.environ.setdefault("HF_HOME", "/data/oujie/models/hf_home")
# os.environ.setdefault("HF_HOME", "/home/models/oujie-data/hf_home")
# Merge: 修改 HF_DATASETS_CACHE 路径为共享存储路径
Original: os.environ.setdefault("HF_DATASETS_CACHE", "/data/oujie/models/hf_home/datasets")
# os.environ.setdefault("HF_DATASETS_CACHE", "/home/models/oujie-data/hf_home/datasets")
# Merge: 添加 HF_TOKEN 设置
# Original: # os.environ.setdefault("HF_TOKEN", "YOUR_HF_TOKEN") # 请通过环境变量 HF_TOKEN 设置
# 从环境变量获取 HF_TOKEN，如果未设置则不设置
# os.environ.setdefault("HF_TOKEN", "YOUR_HF_TOKEN")  # 请通过环境变量 HF_TOKEN 设置

# 添加根路径以定位 util 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from util.dataset_handlers import get_dataset_handler

from config import AdaCacheConfig, create_config_from_args
from manyshot_kv_core import ManyShotKVEvaluator
from model_utils import need_trust_remote_code
from generate_manyshot_summary import generate_summary_from_manyshot_results


def safe_model_id(model_name: str) -> str:
    """将模型名称转换为安全的文件名"""
    return model_name.rstrip("/").split("/")[-1]


def _detect_device() -> str:
    """检测可用设备"""
    if os.environ.get("FORCE_NPU", "").lower() in ("1", "true", "yes"):
        return "npu"
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, model_root: str, device_kind: str, use_sharding: bool = True):
    """加载模型和分词器"""
    if os.path.isabs(model_name):
        local_path = model_name
    else:
        local_path = os.path.join(model_root, model_name)
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"模型路径不存在: {local_path}")
    
    logging.info(f"加载模型: {local_path}")
    
    need_trust_remote = need_trust_remote_code(model_name)
    if "pangu" in model_name.lower():
        need_trust_remote = True
    
    logging.info(f"模型类型检测: {'需要' if need_trust_remote else '不需要'} trust_remote_code")
    # Merge: 移除 local_files_only=True 以支持在线下载
    # Original: tokenizer = AutoTokenizer.from_pretrained(
    #     local_path,
    #     trust_remote_code=need_trust_remote,
    #     use_fast=True,
    #     local_files_only=True
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        local_path, 
        trust_remote_code=need_trust_remote, 
        use_fast=True
    )
    
    dtype = torch.float16 if device_kind in ("cuda", "npu") else None
    
    npu_cnt = 0
    if device_kind == "npu":
        try:
            npu_cnt = torch.npu.device_count()
        except Exception:
            npu_cnt = 1
    
    use_sharding = (device_kind == "npu" and npu_cnt > 1 and use_sharding)
    
    if use_sharding:
        logging.info(f"使用分片模式加载模型 (device_map=auto)")
        # Merge: 移除 local_files_only=True 以支持在线下载
        # Original: model = AutoModelForCausalLM.from_pretrained(
        #     local_path,
        #     trust_remote_code=need_trust_remote,
        #     torch_dtype=dtype,
        #     low_cpu_mem_usage=True,
        #     device_map="auto",
        #     local_files_only=True
        # )
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            trust_remote_code=need_trust_remote,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        logging.info(f"模型分片情况: {getattr(model, 'hf_device_map', 'N/A')}")
    else:
        target = "npu" if device_kind == "npu" else ("cuda" if device_kind == "cuda" else "cpu")
        logging.info(f"加载模型到 {target}")
        # Merge: 移除 local_files_only=True 以支持在线下载
        # Original: model = AutoModelForCausalLM.from_pretrained(
        #     local_path,
        #     trust_remote_code=need_trust_remote,
        #     torch_dtype=dtype,
        #     low_cpu_mem_usage=True,
        #     local_files_only=True
        # )
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            trust_remote_code=need_trust_remote,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        model.to(target)
    
    model.eval()
    
    if device_kind == "cuda":
        logging.info(f"使用 CUDA (GPU 数量: {torch.cuda.device_count()})")
    elif device_kind == "npu":
        logging.info(f"使用 NPU (NPU 数量: {npu_cnt})")
    else:
        logging.info("使用 CPU")
    
    return tokenizer, model


# Merge: 添加 tasks 参数支持
# Original: def run_manyshot_kv_experiment(
#     config: AdaCacheConfig,
#     model_name: str,
#     dataset_name: str,
#     dataset_subset: str = None
# ):
def run_manyshot_kv_experiment(
    config: AdaCacheConfig,
    model_name: str,
    dataset_name: str,
    dataset_subset: str = None,
    tasks: str = None  # 🔥 添加 tasks 参数
):
    """
    运行 Many-Shot KV Cache 实验
    
    Args:
        config: 配置对象
        model_name: 模型名称
        dataset_name: 数据集名称
        dataset_subset: 数据集子集
        tasks: 任务列表（逗号分隔），用于过滤 CoT-Collection 等数据集
    """
    device_kind = config.device or _detect_device()
    
    tokenizer, model = load_model(model_name, config.model_root, device_kind, config.use_sharding)
    
    # Merge: 添加 tasks 参数传递
    # Original: dataset_handler = get_dataset_handler(dataset_name, dataset_subset)
    dataset_handler = get_dataset_handler(dataset_name, dataset_subset, tasks)
    
    logging.info(f"加载测试集: {dataset_name}/{dataset_subset or 'default'}")
    _, test_set = dataset_handler.load_and_split(test_size=config.eval_samples, seed=config.seed)
    logging.info(f"测试集大小: {len(test_set)}")
    
    evaluator = ManyShotKVEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        device=device_kind
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = safe_model_id(model_name)
    # Merge: 添加冒号替换以支持更多格式
    # Original: dataset_id = dataset_name.replace('/', '_')
    dataset_id = dataset_name.replace('/', '_').replace(':', '_')
    if dataset_subset:
        dataset_id += f"_{dataset_subset}"
    
    run_mode = f"manyshot_kv_{dataset_id}"
    run_root = os.path.join(config.output_dir, run_mode, config.run_id)
    os.makedirs(run_root, exist_ok=True)
    
    try:
        mode_root = os.path.join(config.output_dir, run_mode)
        latest_link = os.path.join(mode_root, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            pass
        rel_target = os.path.relpath(run_root, mode_root)
        os.symlink(rel_target, latest_link)
    except Exception:
        pass
    
    file_prefix = f"{model_id}_w{config.window_size}_tau{config.entropy_threshold}_{timestamp}"
    output_prefix = os.path.join(run_root, file_prefix)
    
    logging.info("=" * 80)
    logging.info(f"开始 Many-Shot KV Cache 评估")
    logging.info(f"  模型: {model_name}")
    logging.info(f"  数据集: {dataset_name}/{dataset_subset or 'default'}")
    logging.info(f"  设备: {device_kind}")
    logging.info(f"  全局池大小: {config.global_pool_size}")
    logging.info(f"  窗口大小: {config.window_size}")
    logging.info(f"  熵阈值: {config.entropy_threshold}")
    logging.info("=" * 80)
    
    metrics = evaluator.evaluate(test_set, output_prefix)
    
    del model
    del tokenizer
    if device_kind == "cuda":
        torch.cuda.empty_cache()
    elif device_kind == "npu":
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Many-Shot KV Cache - 基于KV缓存的检索与复用系统")
    
    # 基础参数
    parser.add_argument("--mode", type=str, default="cot", choices=["cot", "io", "paper"], help="模式：cot/io/paper")
    parser.add_argument("--global_pool_size", type=int, default=1024, help="全局示例池大小")
    parser.add_argument("--window_size", type=int, default=4, help="探针窗口大小")
    parser.add_argument("--entropy_threshold", type=float, default=0.5, help="熵阈值")
    parser.add_argument("--max_probe_rounds", type=int, default=256, help="最大探针轮数")
    
    # 模型与数据集
    parser.add_argument("--models", type=str, 
                       default="/data/oujie/models/pangu/FreedomIntelligence/openPangu-Embedded-1B-V1.1",
                       help="模型路径")
    parser.add_argument("--datasets", type=str, default="openai/gsm8k:main",
                       help="数据集配置")
    # Merge: 添加 tasks 参数
    # Original: 无此参数
    parser.add_argument("--tasks", type=str, default=None,
                       help="任务列表（逗号分隔），用于过滤 CoT-Collection 等数据集")
    
    # 实验配置
    parser.add_argument("--eval_samples", type=int, default=100, help="评测样本数")
    parser.add_argument("--gen_tokens", type=int, default=512, help="生成 token 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 路径配置
    parser.add_argument("--model_root", type=str, default="/home/models/models/", help="模型根目录")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    
    # 运行配置
    parser.add_argument("--run_id", type=str, default="manyshot_kv_test", help="运行 ID")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    
    args = parser.parse_args()
    
    log_level = logging.INFO
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "logs", f"manyshot_kv_{args.run_id}.log"))
        ]
    )
    
    config = create_config_from_args(args)
    
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    
    datasets = []
    for ds in args.datasets.split(','):
        ds = ds.strip()
        if ':' in ds:
            name, subset = ds.split(':', 1)
            datasets.append({"name": name.strip(), "subset": subset.strip()})
        else:
            datasets.append({"name": ds, "subset": None})
    
    logging.info("=" * 80)
    logging.info("Many-Shot KV Cache 实验配置")
    logging.info("=" * 80)
    logging.info(f"模型数量: {len(models)}")
    logging.info(f"数据集数量: {len(datasets)}")
    logging.info(f"全局池大小: {config.global_pool_size}")
    logging.info(f"窗口大小: {config.window_size}")
    logging.info(f"熵阈值: {config.entropy_threshold}")
    logging.info(f"评测样本数: {config.eval_samples}")
    logging.info("=" * 80)
    
    all_results = []
    
    for model_name in models:
        for dataset_config in datasets:
            dataset_name = dataset_config["name"]
            dataset_subset = dataset_config["subset"]
            
            try:
                logging.info("\n\n")
                logging.info("#" * 80)
                logging.info(f"# 实验: {model_name} × {dataset_name}/{dataset_subset or 'default'}")
                logging.info("#" * 80)
                
                # Merge: 传递 tasks 参数
                # Original: metrics = run_manyshot_kv_experiment(
                #     config=config,
                #     model_name=model_name,
                #     dataset_name=dataset_name,
                #     dataset_subset=dataset_subset
                # )
                metrics = run_manyshot_kv_experiment(
                    config=config,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    dataset_subset=dataset_subset,
                    tasks=args.tasks  # 🔥 传递 tasks 参数
                )
                
                all_results.append({
                    "model": model_name,
                    "dataset": f"{dataset_name}/{dataset_subset or 'default'}",
                    "metrics": metrics
                })
                
            except Exception as e:
                logging.error(f"实验失败: {e}", exc_info=True)
    
    logging.info("\n\n")
    logging.info("=" * 80)
    logging.info("所有实验汇总")
    logging.info("=" * 80)
    for result in all_results:
        logging.info(f"\n模型: {result['model']}")
        logging.info(f"数据集: {result['dataset']}")
        logging.info(f"  准确率: {result['metrics']['acc_numeric']:.4f}")
        logging.info(f"  平均 shot 数: {result['metrics']['num_shots_mean']:.2f}")
    
    logging.info("\n✓ 所有实验完成！")

if __name__ == "__main__":
    main()
