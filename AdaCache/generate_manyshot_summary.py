#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Many-Shot KV 的 JSONL 结果文件生成汇总表（支持任意模型）
"""
import os
import json
import csv
import numpy as np
from pathlib import Path


def analyze_jsonl_results(jsonl_path):
    """
    分析 JSONL 结果文件，提取汇总指标
    
    Args:
        jsonl_path: JSONL 文件路径
    
    Returns:
        dict: 汇总指标
    """
    results = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        return None
    
    # 统计指标
    num_shots_list = [r['num_shots'] for r in results]
    em_string_list = [r['EM_string'] for r in results]
    contains_list = [r['Contains'] for r in results]
    token_f1_list = [r['Token_F1'] for r in results]
    numeric_em_list = [r['Numeric_EM'] for r in results]
    
    # 相对误差（过滤 None）
    relerr_list = [r['rel_error'] for r in results if r['rel_error'] is not None]
    
    # 时间和 token 统计
    time_list = [r['time_spent'] for r in results]
    kv_tokens_list = [r['kv_tokens'] for r in results]
    output_tokens_list = [r['output_tokens'] for r in results]
    
    # 计算吞吐量
    tokpersec_list = []
    for r in results:
        if r['time_spent'] > 0 and r['output_tokens'] > 0:
            tokpersec_list.append(r['output_tokens'] / r['time_spent'])
    
    summary = {
        'count': len(results),
        'num_shots_mean': float(np.mean(num_shots_list)),
        'num_shots_median': float(np.median(num_shots_list)),
        'num_shots_min': int(np.min(num_shots_list)),
        'num_shots_max': int(np.max(num_shots_list)),
        'acc_numeric': float(np.mean(numeric_em_list)),
        'em_string': float(np.mean(em_string_list)),
        'contains': float(np.mean(contains_list)),
        'token_f1': float(np.mean(token_f1_list)),
        'relerr_mean': float(np.mean(relerr_list)) if relerr_list else 0.0,
        'relerr_median': float(np.median(relerr_list)) if relerr_list else 0.0,
        'total_time_s': float(np.sum(time_list)),  # 🔥 新增：总运行时间
        'avg_latency_s': float(np.mean(time_list)),
        'avg_kv_tokens': float(np.mean(kv_tokens_list)),
        'avg_out_tokens': float(np.mean(output_tokens_list)),
        'avg_tokpersec': float(np.mean(tokpersec_list)) if tokpersec_list else 0.0
    }
    
    return summary


def parse_filename(filename):
    """
    从文件名解析配置信息
    支持多种文件名格式：
    - Qwen2.5-7B_w4_tau5.0_20251203_151042.jsonl
    - Meta-Llama-3.1-70B_w8_tau0.3_timestamp.jsonl
    - model_name_wX_tauY.Y_timestamp.jsonl
    
    Returns:
        dict: 包含 model, window_size, entropy_threshold 的字典
    """
    parts = filename.replace('.jsonl', '').split('_')
    
    model_parts = []
    window_size = None
    entropy_threshold = None
    
    for i, part in enumerate(parts):
        if part.startswith('w') and len(part) <= 4 and part[1:].replace('.', '').isdigit():
            # 窗口大小：w4, w8, w16 等
            if window_size is None:
                try:
                    window_size = int(part[1:])
                except:
                    pass
        elif part.startswith('tau'):
            # 熵阈值：tau0.3, tau5.0 等
            try:
                entropy_threshold = float(part.replace('tau', ''))
            except:
                pass
        elif not part.isdigit() and len(part) != 8:  # 排除时间戳
            # 模型名称部分
            model_parts.append(part)
    
    # 拼接模型名称
    model_name = '_'.join(model_parts) if model_parts else 'unknown'
    
    return {
        'model': model_name,
        'window_size': window_size or 0,
        'entropy_threshold': entropy_threshold or 0.0
    }


def infer_dataset_from_path(file_path):
    """
    从文件路径推断数据集和任务信息
    
    Args:
        file_path: Path 对象
    
    Returns:
        tuple: (dataset_name, subset, task_name)
    """
    path_str = str(file_path).lower()
    
    # 常见数据集匹配
    dataset_mapping = {
        'gsm8k': ('openai/gsm8k', 'main'),
        'aqua': ('aqua_rat', 'raw'),
        'math500': ('math500', 'default'),
        'svamp': ('svamp', 'default'),
        'asdiv': ('asdiv', 'default'),
        'mawps': ('mawps', 'default'),
        'cot-collection': ('cot-collection', 'default'),
        'ugphysics': ('UGPhysics/ugphysics', 'mixed')
    }
    
    # 从路径中提取任务名
    task_name = 'unknown'
    # 查找路径中的任务标识符
    import re
    # 匹配类似 llama_3.2_3b_taskname 的模式
    match = re.search(r'llama_3[._]2[._]3b[._]([^/\\]+)', path_str)
    if match:
        task_name = match.group(1)
    
    # 根据任务名确定数据集
    dataset = 'unknown'
    subset = 'unknown'
    for key, (ds, sb) in dataset_mapping.items():
        if key in task_name:
            dataset = ds
            subset = sb
            break
    
    # 如果没找到匹配的数据集，尝试从路径中推断
    if dataset == 'unknown':
        for key, (ds, sb) in dataset_mapping.items():
            if key in path_str:
                dataset = ds
                subset = sb
                break
    
    return dataset, subset, task_name


def generate_summary_from_manyshot_results(results_dir, output_csv, run_id='manyshot_kv'):
    """
    从 Many-Shot KV 的结果目录生成汇总表（支持任意模型）
    
    Args:
        results_dir: 结果目录路径
        output_csv: 输出 CSV 路径
        run_id: 运行 ID
    """
    # 查找所有 JSONL 文件（排除 probe_details）
    jsonl_files = [f for f in Path(results_dir).glob('**/*.jsonl') 
                   if 'probe_details' not in f.name]
    
    if not jsonl_files:
        print(f"在 {results_dir} 中没有找到任何 JSONL 文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个 JSONL 文件")
    
    # CSV 表头
    headers = [
        "run_id", "mode", "dataset", "subset", "task_name", "model",
        "global_pool_size", "entropy_threshold", "window_size", "paper_k_full",
        "count", "optimal_k_mean", "optimal_k_median", "optimal_k_min", "optimal_k_max",
        "acc_numeric", "em_string", "contains", "token_f1",
        "acc_tol_1e-4", "acc_tol_1e-3", "acc_tol_1e-2",
        "relerr_mean", "relerr_median",
        "total_time_s", "avg_latency_s", "avg_in_tokens", "avg_out_tokens", "avg_tokpersec"  # 🔥 新增 total_time_s
    ]
    
    all_rows = []
    
    for jsonl_file in jsonl_files:
        print(f"\n处理: {jsonl_file.name}")
        
        # 分析结果
        summary = analyze_jsonl_results(str(jsonl_file))
        if summary is None:
            print(f"  ✗ 文件为空或无效")
            continue
        
        # 解析文件名
        config = parse_filename(jsonl_file.name)
        
        # 从路径推断数据集和任务
        dataset, subset, task_name = infer_dataset_from_path(jsonl_file)
        
        # 尝试读取同前缀的 metrics.json 以获取真实配置
        metrics_path = str(jsonl_file).replace('.jsonl', '_metrics.json')
        metrics = None
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r', encoding='utf-8') as mf:
                    metrics = json.load(mf)
            except Exception:
                metrics = None

        gp_size = 100
        win_size = config['window_size']
        tau = config['entropy_threshold']
        if metrics:
            gp_size = metrics.get('global_pool_size', gp_size)
            win_size = metrics.get('window_size', win_size)
            tau = metrics.get('entropy_threshold', tau)

        # 构建行
        row = {
            "run_id": run_id,
            "mode": "manyshot_kv",
            "dataset": dataset,
            "subset": subset,
            "task_name": task_name,
            "model": config['model'],
            "global_pool_size": gp_size,
            "entropy_threshold": f"{float(tau):.2f}",
            "window_size": win_size,
            "paper_k_full": 0,
            "count": summary['count'],
            "optimal_k_mean": f"{summary['num_shots_mean']:.2f}",
            "optimal_k_median": f"{summary['num_shots_median']:.2f}",
            "optimal_k_min": summary['num_shots_min'],
            "optimal_k_max": summary['num_shots_max'],
            "acc_numeric": f"{summary['acc_numeric']:.4f}",
            "em_string": f"{summary['em_string']:.4f}",
            "contains": f"{summary['contains']:.4f}",
            "token_f1": f"{summary['token_f1']:.4f}",
            "acc_tol_1e-4": "0.0000",
            "acc_tol_1e-3": "0.0000",
            "acc_tol_1e-2": "0.0000",
            "relerr_mean": f"{summary['relerr_mean']:.4f}",
            "relerr_median": f"{summary['relerr_median']:.4f}",
            "total_time_s": f"{summary['total_time_s']:.2f}",  # 🔥 新增
            "avg_latency_s": f"{summary['avg_latency_s']:.2f}",
            "avg_in_tokens": f"{summary['avg_kv_tokens']:.1f}",
            "avg_out_tokens": f"{summary['avg_out_tokens']:.1f}",
            "avg_tokpersec": f"{summary['avg_tokpersec']:.2f}"
        }
        
        all_rows.append(row)
        
        # 打印摘要
        print(f"  ✓ {config['model']}")
        print(f"    准确率: {summary['acc_numeric']:.2%}")
        print(f"    平均 shots: {summary['num_shots_mean']:.1f}")
        print(f"    平均输出: {summary['avg_out_tokens']:.1f} tokens")
    
    # 写入 CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    
    print(f"\n✓ 汇总表已生成: {output_csv}")
    print(f"  共 {len(all_rows)} 条记录")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从 Many-Shot KV JSONL 结果生成汇总表（支持任意模型）")
    # Merge: 修改默认路径为共享存储路径
    # Original: parser.add_argument("--results_dir", type=str, 
    # Original:                    default="./outputs",
    # Original:                    help="结果目录（会递归查找所有 JSONL 文件）")
    parser.add_argument("--results_dir", type=str, 
                       default="/data/oujie/oujie-data/shareShot/AdaCache",
                       help="结果目录（会递归查找所有 JSONL 文件）")
    # Merge: 修改默认输出路径为共享存储路径
    # Original: parser.add_argument("--output_csv", type=str,
    # Original:                    default="./outputs/summary_manyshot_kv.csv",
    # Original:                    help="输出 CSV 文件路径")
    parser.add_argument("--output_csv", type=str,
                       default="/data/oujie/oujie-data/shareShot/AdaCache/summary_manyshot_kv.csv",
                       help="输出 CSV 文件路径")
    parser.add_argument("--run_id", type=str, default="manyshot_kv", help="运行 ID")
    
    args = parser.parse_args()
    
    generate_summary_from_manyshot_results(args.results_dir, args.output_csv, args.run_id)
