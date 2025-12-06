"""
Shot Baseline 主运行脚本
支持三种模式：CoT, IO, Paper
"""
import sys
import os
import argparse
import logging
import torch
import csv
import random
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "60")
os.environ.setdefault("HF_HOME", "/data/oujie/models/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/oujie/models/hf_home/datasets")
os.environ.setdefault("HF_TOKEN", "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")

# 添加 util 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# 添加 AdaCache 路径（复用 model_utils）
adacache_path = os.path.join(os.path.dirname(__file__), '../AdaCache')
sys.path.append(adacache_path)

from transformers import AutoTokenizer, AutoModelForCausalLM
from util.dataset_handlers import get_dataset_handler

# 从当前目录导入 config 和 shot_core
import importlib.util
config_path = os.path.join(os.path.dirname(__file__), 'config.py')
spec = importlib.util.spec_from_file_location("shot_config", config_path)
shot_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shot_config_module)
ShotConfig = shot_config_module.ShotConfig
create_config_from_args = shot_config_module.create_config_from_args

shot_core_path = os.path.join(os.path.dirname(__file__), 'shot_core.py')
spec = importlib.util.spec_from_file_location("shot_core_module", shot_core_path)
shot_core_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shot_core_module)
ShotEvaluator = shot_core_module.ShotEvaluator

# 从 AdaCache 导入 model_utils
sys.path.insert(0, adacache_path)
from model_utils import need_trust_remote_code


def safe_model_id(model_name: str) -> str:
    """将模型名称转换为安全的文件名"""
    return model_name.rstrip("/").split("/")[-1]


def write_summary_tables(all_results, output_path, run_id):
    """
    生成汇总表（CSV 和 Markdown 格式）
    支持追加模式：如果文件已存在，会读取现有数据并追加新结果（避免重复）
    
    Args:
        all_results: 所有实验结果列表
        output_path: 输出路径（汇总表将保存在此路径下）
        run_id: 运行 ID
    """
    if not all_results:
        logging.warning("没有实验结果，跳过汇总表生成")
        return
    
    # 直接在指定路径下创建汇总表
    csv_path = os.path.join(output_path, f"summary_{run_id}.csv")
    md_path = os.path.join(output_path, f"summary_{run_id}.md")
    
    # 定义表头
    headers = [
        "run_id", "mode", "dataset", "subset", "model",
        "avg_num_shots",
        "count", "acc_numeric", "em_string", "contains", "token_f1",
        "acc_tol_1e-4", "acc_tol_1e-3", "acc_tol_1e-2",
        "relerr_mean", "relerr_median",
        "total_time_s", "avg_latency_s", "avg_in_tokens", "avg_out_tokens", "avg_tokpersec"
    ]
    
    # 准备新数据
    new_rows = []
    for result in all_results:
        model_id = safe_model_id(result['model'])
        metrics = result['metrics']
        
        row = {
            "run_id": run_id,
            "mode": result.get('mode', 'unknown'),
            "dataset": result['dataset_name'],
            "subset": result['dataset_subset'],
            "model": model_id,
            "avg_num_shots": f"{metrics.get('avg_num_shots', 0):.1f}",
            "count": metrics.get('count', 0),
            "acc_numeric": f"{metrics.get('acc_numeric', 0):.4f}",
            "em_string": f"{metrics.get('em_string', 0):.4f}",
            "contains": f"{metrics.get('contains', 0):.4f}",
            "token_f1": f"{metrics.get('token_f1', 0):.4f}",
            "acc_tol_1e-4": f"{metrics.get('acc_tol_1e-4', 0):.4f}",
            "acc_tol_1e-3": f"{metrics.get('acc_tol_1e-3', 0):.4f}",
            "acc_tol_1e-2": f"{metrics.get('acc_tol_1e-2', 0):.4f}",
            "relerr_mean": f"{metrics.get('relerr_mean', 0):.6f}",
            "relerr_median": f"{metrics.get('relerr_median', 0):.6f}",
            "total_time_s": f"{metrics.get('total_time_s', 0):.3f}",
            "avg_latency_s": f"{metrics.get('avg_latency_s', 0):.3f}",
            "avg_in_tokens": f"{metrics.get('avg_in_tokens', 0):.1f}",
            "avg_out_tokens": f"{metrics.get('avg_out_tokens', 0):.1f}",
            "avg_tokpersec": f"{metrics.get('avg_tokpersec', 0):.2f}"
        }
        new_rows.append(row)
    
    # 读取现有数据（如果存在）
    existing_rows = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
            logging.info(f"读取到 {len(existing_rows)} 条现有记录")
        except Exception as e:
            logging.warning(f"读取现有CSV文件失败: {e}，将创建新文件")
            existing_rows = []
    
    # 去重：基于模型、数据集、模式的组合
    def make_key(row):
        """生成唯一键用于去重"""
        return (
            row.get('run_id', ''),
            row.get('model', ''),
            row.get('dataset', ''),
            row.get('subset', ''),
            row.get('mode', ''),
        )
    
    # 将现有数据转换为字典（用于去重）
    existing_dict = {make_key(row): row for row in existing_rows}
    
    # 添加新数据（如果键已存在则覆盖，否则添加）
    for new_row in new_rows:
        key = make_key(new_row)
        existing_dict[key] = new_row
    
    # 合并后的所有数据
    all_rows = list(existing_dict.values())
    
    logging.info(f"合并后共 {len(all_rows)} 条记录（新增/更新 {len(new_rows)} 条）")
    
    # 写入 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)
    
    # 写入 Markdown（表格格式）
    with open(md_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write(f"# Shot Baseline 实验汇总\n\n")
        f.write(f"运行 ID: {run_id}\n")
        f.write(f"总记录数: {len(all_rows)}\n\n")
        
        # 写入表格
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for row in all_rows:
            row_values = [str(row.get(h, '')) for h in headers]
            f.write("| " + " | ".join(row_values) + " |\n")
    
    logging.info(f"\n汇总表已保存:")
    logging.info(f"  CSV: {csv_path}")
    logging.info(f"  Markdown: {md_path}")


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
    # 确定本地路径
    if os.path.isabs(model_name):
        local_path = model_name
    else:
        local_path = os.path.join(model_root, model_name)
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"模型路径不存在: {local_path}")
    
    logging.info(f"加载模型: {local_path}")
    
    # 检测模型类型
    need_trust_remote = need_trust_remote_code(model_name)
    
    # === 新增：强制 openPangu 使用远程代码 ===
    if "pangu" in model_name.lower():
        need_trust_remote = True
    # ========================================
    # 加载分词器
    logging.info(f"模型类型检测: {'需要' if need_trust_remote else '不需要'} trust_remote_code")
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        trust_remote_code=need_trust_remote,
        local_files_only=True
    )
    
    # 加载模型
    if device_kind == "npu":
        if use_sharding:
            npu_cnt = torch.npu.device_count()
            logging.info(f"使用 NPU 自动分片 ({npu_cnt} 张卡)")
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                trust_remote_code=need_trust_remote,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                trust_remote_code=need_trust_remote,
                torch_dtype=torch.float16,
                local_files_only=True
            )
            model.to("npu:0")
    elif device_kind == "cuda":
        if use_sharding:
            gpu_cnt = torch.cuda.device_count()
            logging.info(f"使用 CUDA 自动分片 ({gpu_cnt} 张卡)")
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                trust_remote_code=need_trust_remote,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                trust_remote_code=need_trust_remote,
                torch_dtype=torch.float16,
                local_files_only=True
            )
            model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            trust_remote_code=need_trust_remote,
            local_files_only=True
        )
        target = torch.device("cpu")
        model.to(target)
    
    model.eval()
    
    # 输出设备信息
    if device_kind == "cuda":
        logging.info(f"使用 CUDA (GPU 数量: {torch.cuda.device_count()})")
    elif device_kind == "npu":
        npu_cnt = torch.npu.device_count() if hasattr(torch, 'npu') else 0
        logging.info(f"使用 NPU (NPU 数量: {npu_cnt})")
    else:
        logging.info("使用 CPU")
    
    return tokenizer, model


def run_shot_experiment(
    config: ShotConfig,
    model_name: str,
    dataset_name: str,
    dataset_subset: str = None,
    tasks: str = None
):
    """
    运行单个 Shot 实验
    
    Args:
        config: Shot 配置
        model_name: 模型名称
        dataset_name: 数据集名称
        dataset_subset: 数据集子集
        tasks: 任务列表（逗号分隔）
    """
    # 检测设备
    if config.device and config.device.strip():
        device_kind = config.device
    else:
        device_kind = _detect_device()
    
    # 加载模型
    tokenizer, model = load_model(model_name, config.model_root, device_kind, config.use_sharding)
    
    # 获取数据集处理器
    dataset_handler = get_dataset_handler(dataset_name, dataset_subset, tasks)
    
    # 加载数据集
    logging.info(f"加载数据集: {dataset_name}/{dataset_subset or 'default'}")
    if tasks:
        logging.info(f"任务过滤: {tasks}")
    
    # load_and_split 只接受 test_size 和 seed 参数
    # 训练集将是剩余的所有数据
    train_set, test_set = dataset_handler.load_and_split(
        test_size=config.eval_samples, 
        seed=config.seed
    )
    
    logging.info(f"训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")
    
    # 设置随机种子
    random.seed(config.seed)
    
    # 创建评估器
    evaluator = ShotEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dataset_handler=dataset_handler,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        train_pool=train_set
    )
    
    # 创建输出目录结构: outputs/shot_dataset_mode/run_id/model_name/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = safe_model_id(model_name)
    dataset_id = dataset_name.replace('/', '_')
    if dataset_subset:
        dataset_id += f"_{dataset_subset}"
    
    run_mode = f"shot_{dataset_id}_{config.mode}"
    run_root = os.path.join(config.output_dir, run_mode, config.run_id, model_id)
    os.makedirs(run_root, exist_ok=True)
    
    # 创建 latest 软链接
    try:
        mode_root = os.path.join(config.output_dir, run_mode)
        latest_link = os.path.join(mode_root, "latest")
        run_id_path = os.path.join(config.output_dir, run_mode, config.run_id)
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            pass
        rel_target = os.path.relpath(run_id_path, mode_root)
        os.symlink(rel_target, latest_link)
    except Exception:
        pass
    
    # 文件命名
    if config.mode == "paper":
        file_prefix = f"{model_id}_{config.mode}_k{config.paper_k_full}_q{config.paper_num_questions}_{timestamp}"
    else:
        file_prefix = f"{model_id}_{config.mode}_{config.num_shots}shot_{timestamp}"
    
    output_prefix = os.path.join(run_root, file_prefix)
    
    # 运行评估
    logging.info("=" * 80)
    logging.info(f"开始 Shot Baseline 评估")
    logging.info(f"  模型: {model_name}")
    logging.info(f"  数据集: {dataset_name}/{dataset_subset or 'default'}")
    logging.info(f"  模式: {config.mode.upper()}")
    if config.mode == "cot" or config.mode == "io":
        logging.info(f"  Shot 数: {config.num_shots}")
    else:
        logging.info(f"  Fullshot 数: {config.paper_k_full}")
        logging.info(f"  Question-only 数: {config.paper_num_questions}")
    logging.info("=" * 80)
    
    metrics = evaluator.evaluate(test_set, output_prefix)
    
    # 释放显存
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
    parser = argparse.ArgumentParser(description="Shot Baseline 评估")
    
    # 模型与数据集
    parser.add_argument("--models", type=str, required=True,
                       help="模型路径（多个用逗号分隔）")
    parser.add_argument("--datasets", type=str, required=True,
                       help="数据集配置（格式：dataset1:subset1,dataset2:subset2）")
    
    # Shot 模式配置
    parser.add_argument("--mode", type=str, default="cot", choices=["cot", "io", "paper"],
                       help="Shot 模式：cot (Chain-of-Thought), io (Input-Output), paper")
    parser.add_argument("--num_shots", type=int, default=4,
                       help="CoT/IO 模式下的 shot 数量")
    parser.add_argument("--paper_k_full", type=int, default=4,
                       help="Paper 模式下的 fullshot 数量")
    parser.add_argument("--paper_num_questions", type=int, default=4,
                       help="Paper 模式下的 question-only shots 数量")
    
    # 实验配置
    parser.add_argument("--eval_samples", type=int, default=100, help="评测样本数")
    parser.add_argument("--gen_tokens", type=int, default=512, help="生成 token 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tasks", type=str, default=None, help="按逗号分隔的任务列表，用于过滤数据集样本")
    
    # 路径配置
    parser.add_argument("--model_root", type=str, default="/home/models/models/", help="模型根目录")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    
    # 运行配置
    parser.add_argument("--run_id", type=str, default="shot_test", help="运行 ID")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.INFO if args.verbose else logging.INFO
    
    # 创建日志目录
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志系统
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, f"shot_{args.mode}_{args.run_id}.log"))
        ]
    )
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 解析模型列表
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    
    # 解析数据集列表
    datasets = []
    for ds in args.datasets.split(','):
        ds = ds.strip()
        if ':' in ds:
            name, subset = ds.split(':', 1)
            datasets.append({"name": name.strip(), "subset": subset.strip()})
        else:
            datasets.append({"name": ds, "subset": None})
    
    # 运行实验
    logging.info("=" * 80)
    logging.info("Shot Baseline 实验配置")
    logging.info("=" * 80)
    logging.info(f"模型数量: {len(models)}")
    logging.info(f"数据集数量: {len(datasets)}")
    logging.info(f"模式: {config.mode.upper()}")
    if config.mode == "cot" or config.mode == "io":
        logging.info(f"Shot 数: {config.num_shots}")
    else:
        logging.info(f"Fullshot 数: {config.paper_k_full}")
        logging.info(f"Question-only 数: {config.paper_num_questions}")
    logging.info(f"评测样本数: {config.eval_samples}")
    logging.info("=" * 80)
    
    # 按数据集组织结果
    dataset_results = {}
    
    for model_name in models:
        for dataset_config in datasets:
            dataset_name = dataset_config["name"]
            dataset_subset = dataset_config["subset"]
            
            # 确定数据集标识
            dataset_id = dataset_name.replace('/', '_')
            if dataset_subset:
                dataset_id += f"_{dataset_subset}"
            
            # 为每个数据集+模式组合初始化结果存储
            dataset_mode_id = f"{dataset_id}_{config.mode}"
            if dataset_mode_id not in dataset_results:
                run_mode = f"shot_{dataset_id}_{config.mode}"
                summary_root = os.path.join(config.output_dir, run_mode, config.run_id)
                os.makedirs(summary_root, exist_ok=True)
                dataset_results[dataset_mode_id] = {
                    "results": [],
                    "summary_root": summary_root
                }
            
            try:
                logging.info("\n\n")
                logging.info("#" * 80)
                logging.info(f"# 实验: {model_name} × {dataset_name}/{dataset_subset or 'default'} × {config.mode.upper()}")
                logging.info("#" * 80)
                
                metrics = run_shot_experiment(
                    config=config,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    dataset_subset=dataset_subset,
                    tasks=config.tasks
                )
                
                # 将结果添加到对应的数据集
                dataset_results[dataset_mode_id]["results"].append({
                    "model": model_name,
                    "dataset": f"{dataset_name}/{dataset_subset or 'default'}",
                    "dataset_name": dataset_name,
                    "dataset_subset": dataset_subset or 'default',
                    "mode": config.mode,
                    "metrics": metrics
                })
                
            except Exception as e:
                logging.error(f"实验失败: {e}", exc_info=True)
    
    # 汇总结果
    logging.info("\n\n")
    logging.info("=" * 80)
    logging.info("所有实验汇总")
    logging.info("=" * 80)
    
    # 为每个数据集+模式组合生成汇总表
    for dataset_mode_id, data in dataset_results.items():
        results = data["results"]
        summary_root = data["summary_root"]
        
        if not results:
            logging.warning(f"数据集+模式 {dataset_mode_id} 没有实验结果")
            continue
        
        logging.info(f"\n数据集+模式: {dataset_mode_id}")
        for result in results:
            logging.info(f"  模型: {result['model']}")
            logging.info(f"    准确率: {result['metrics']['acc_numeric']:.4f}")
            logging.info(f"    平均 Shot 数: {result['metrics'].get('avg_num_shots', 0):.1f}")
        
        # 生成汇总表（CSV 和 Markdown）
        write_summary_tables(results, summary_root, args.run_id)
    
    logging.info("\n✓ 所有实验完成！")


if __name__ == "__main__":
    main()
