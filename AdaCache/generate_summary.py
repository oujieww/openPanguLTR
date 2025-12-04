#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为已有的 AdaCache 实验结果生成汇总表
"""
import os
import sys
import json
import csv
import glob
from pathlib import Path

# 添加当前路径
sys.path.insert(0, os.path.dirname(__file__))

def safe_model_id(model_name: str) -> str:
    """将模型名称转换为安全的文件名"""
    return model_name.rstrip("/").split("/")[-1]


def generate_summary_from_results(output_dir, run_id=None):
    """
    从已有的实验结果生成汇总表
    
    Args:
        output_dir: 输出根目录
        run_id: 运行 ID（可选，默认为 'summary'）
    """
    if run_id is None:
        run_id = "summary"
    
    # 查找所有的 metrics.json 文件
    metrics_files = glob.glob(os.path.join(output_dir, "**/*_metrics.json"), recursive=True)
    
    if not metrics_files:
        print(f"在 {output_dir} 中没有找到任何 metrics.json 文件")
        return
    
    print(f"找到 {len(metrics_files)} 个实验结果文件")
    
    all_results = []
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # 从文件名解析配置信息
            filename = os.path.basename(metrics_file)
            # 文件名格式: model_tauX.X_wX_timestamp_metrics.json
            parts = filename.replace("_metrics.json", "").split("_")
            
            # 提取模型名称（第一部分到 tau 之前）
            model_parts = []
            for i, part in enumerate(parts):
                if part.startswith("tau"):
                    break
                model_parts.append(part)
            model_name = "_".join(model_parts)
            
            # 提取参数
            entropy_threshold = None
            window_size = None
            for part in parts:
                if part.startswith("tau"):
                    try:
                        entropy_threshold = float(part.replace("tau", ""))
                    except:
                        pass
                elif part.startswith("w") and len(part) <= 3:
                    try:
                        window_size = int(part.replace("w", ""))
                    except:
                        pass
            
            # 从 metrics 中提取数据集信息
            dataset_full = metrics.get('dataset', 'unknown')
            if '/' in dataset_full:
                parts = dataset_full.split('/')
                dataset_name = parts[0]
                dataset_subset = '/'.join(parts[1:])
            else:
                dataset_name = dataset_full
                dataset_subset = 'default'
            
            result = {
                'model': model_name,
                'dataset': dataset_full,
                'dataset_name': dataset_name,
                'dataset_subset': dataset_subset,
                'metrics': metrics,
                'config': {
                    'global_pool_size': metrics.get('global_pool_size', 0),
                    'entropy_threshold': entropy_threshold or metrics.get('entropy_threshold', 0),
                    'window_size': window_size or metrics.get('window_size', 0),
                    'paper_k_full': metrics.get('paper_k_full', 0)
                }
            }
            
            all_results.append(result)
            print(f"  ✓ {model_name} × {dataset_full}")
            
        except Exception as e:
            print(f"  ✗ 解析失败: {metrics_file} - {e}")
    
    if not all_results:
        print("没有成功解析的实验结果")
        return
    
    # 创建汇总表
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    csv_path = os.path.join(summary_dir, f"summary_{run_id}.csv")
    md_path = os.path.join(summary_dir, f"summary_{run_id}.md")
    
    # 定义表头
    headers = [
        "run_id", "mode", "dataset", "subset", "model",
        "global_pool_size", "entropy_threshold", "window_size", "paper_k_full",
        "count", "optimal_k_mean", "optimal_k_median", "optimal_k_min", "optimal_k_max",
        "acc_numeric", "em_string", "contains", "token_f1",
        "acc_tol_1e-4", "acc_tol_1e-3", "acc_tol_1e-2",
        "relerr_mean", "relerr_median",
        "avg_latency_s", "avg_in_tokens", "avg_out_tokens", "avg_tokpersec"
    ]
    
    # 写入 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for result in all_results:
            model_id = safe_model_id(result['model'])
            metrics = result['metrics']
            config = result['config']
            
            row = {
                "run_id": run_id,
                "mode": "adacache",
                "dataset": result['dataset_name'],
                "subset": result['dataset_subset'],
                "model": model_id,
                "global_pool_size": config['global_pool_size'],
                "entropy_threshold": config['entropy_threshold'],
                "window_size": config['window_size'],
                "paper_k_full": config['paper_k_full'],
                "count": metrics.get('count', 0),
                "optimal_k_mean": f"{metrics.get('optimal_k_mean', 0):.2f}",
                "optimal_k_median": f"{metrics.get('optimal_k_median', 0):.2f}",
                "optimal_k_min": metrics.get('optimal_k_min', 0),
                "optimal_k_max": metrics.get('optimal_k_max', 0),
                "acc_numeric": f"{metrics.get('acc_numeric', 0):.4f}",
                "em_string": f"{metrics.get('em_string', 0):.4f}",
                "contains": f"{metrics.get('contains', 0):.4f}",
                "token_f1": f"{metrics.get('token_f1', 0):.4f}",
                "acc_tol_1e-4": f"{metrics.get('acc_tol_1e-4', 0):.4f}",
                "acc_tol_1e-3": f"{metrics.get('acc_tol_1e-3', 0):.4f}",
                "acc_tol_1e-2": f"{metrics.get('acc_tol_1e-2', 0):.4f}",
                "relerr_mean": f"{metrics.get('relerr_mean', 0):.4f}",
                "relerr_median": f"{metrics.get('relerr_median', 0):.4f}",
                "avg_latency_s": f"{metrics.get('avg_latency_s', 0):.2f}",
                "avg_in_tokens": f"{metrics.get('avg_in_tokens', 0):.1f}",
                "avg_out_tokens": f"{metrics.get('avg_out_tokens', 0):.1f}",
                "avg_tokpersec": f"{metrics.get('avg_tokpersec', 0):.2f}"
            }
            writer.writerow(row)
    
    # 写入 Markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# AdaCache 实验汇总\n\n")
        f.write(f"运行 ID: {run_id}\n\n")
        
        # 写入表格
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for result in all_results:
            model_id = safe_model_id(result['model'])
            metrics = result['metrics']
            config = result['config']
            
            row_values = [
                run_id,
                "adacache",
                result['dataset_name'],
                result['dataset_subset'],
                model_id,
                str(config['global_pool_size']),
                f"{config['entropy_threshold']:.2f}",
                str(config['window_size']),
                str(config['paper_k_full']),
                str(metrics.get('count', 0)),
                f"{metrics.get('optimal_k_mean', 0):.2f}",
                f"{metrics.get('optimal_k_median', 0):.2f}",
                str(metrics.get('optimal_k_min', 0)),
                str(metrics.get('optimal_k_max', 0)),
                f"{metrics.get('acc_numeric', 0):.4f}",
                f"{metrics.get('em_string', 0):.4f}",
                f"{metrics.get('contains', 0):.4f}",
                f"{metrics.get('token_f1', 0):.4f}",
                f"{metrics.get('acc_tol_1e-4', 0):.4f}",
                f"{metrics.get('acc_tol_1e-3', 0):.4f}",
                f"{metrics.get('acc_tol_1e-2', 0):.4f}",
                f"{metrics.get('relerr_mean', 0):.4f}",
                f"{metrics.get('relerr_median', 0):.4f}",
                f"{metrics.get('avg_latency_s', 0):.2f}",
                f"{metrics.get('avg_in_tokens', 0):.1f}",
                f"{metrics.get('avg_out_tokens', 0):.1f}",
                f"{metrics.get('avg_tokpersec', 0):.2f}"
            ]
            f.write("| " + " | ".join(row_values) + " |\n")
    
    print(f"\n✓ 汇总表已生成:")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="为已有实验结果生成汇总表")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--run_id", type=str, default="all", help="运行 ID")
    
    args = parser.parse_args()
    
    generate_summary_from_results(args.output_dir, args.run_id)
