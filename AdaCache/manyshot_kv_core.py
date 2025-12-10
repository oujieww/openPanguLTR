"""
Many-Shot KV Cache 核心评估器
整合完整的 KV cache 检索与复用流程

完整流程:
1. 离线 Prefilling: 构建 1024-shot KV cache 池
2. Query 编码: 提取平均 Q 向量 q̄
3. Shot 排序: Token级打分 -> Shot级聚合 -> 排序
4. 探针选择: 按窗口n逐轮扩展，熵判断停止
5. KV 拼装: 将选中shots的KV与prompt+query的KV拼接
6. 最终生成: 使用拼装的KV cache生成答案
"""
import sys
import os
import json
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

# 优先使用本地 AdaCache 的辅助模块
sys.path.insert(0, os.path.dirname(__file__))
# Merge: 修改路径从 ../baseline 到 ../util
# Original: # 再加入 baseline 路径以复用其余组件
# Original: sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../baseline'))
# 加入根路径以复用 util 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from util.dataset_handlers import get_dataset_handler

# 导入评估函数
try:
    from util.new_metrics import evaluate_answer
except ImportError:
    from util.metrics_utils import (
        normalize_answer,
        token_f1_pair,
        parse_number_from_text,
        numeric_equal,
        relative_error
    )
    
    def evaluate_answer(pred_final: str, gold_final: str) -> dict:
        em_str = 1.0 if normalize_answer(pred_final) == normalize_answer(gold_final) else 0.0
        contains = 1.0 if normalize_answer(gold_final) in normalize_answer(pred_final) else 0.0
        tf1 = token_f1_pair(pred_final, gold_final)
        
        gold_num = parse_number_from_text(gold_final)
        pred_num = parse_number_from_text(pred_final)
        
        numeric_ok = 0
        relerr = None
        
        if gold_num is not None and pred_num is not None:
            numeric_ok = 1 if numeric_equal(pred_num, gold_num) else 0
            relerr = relative_error(pred_num, gold_num)
        
        return {
            "numeric_ok": numeric_ok,
            "relerr": relerr,
            "em_str": em_str,
            "contains": contains,
            "tf1": tf1
        }

from bm25_retriever import BM25Retriever
from kv_pool_manager import KVPoolManager
from query_encoder import QueryEncoder
from shot_ranker import ShotRanker
from probe_selector import ProbeSelector
from kv_assembler import KVAssembler
from config import AdaCacheConfig


class ManyShotKVEvaluator:
    """Many-Shot KV Cache 评估器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        config: AdaCacheConfig,
        dataset_name: str,
        dataset_subset: str = None,
        tasks: str = None,
        device: str = "npu"
    ):
        """
        初始化评估器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            config: 配置对象
            dataset_name: 数据集名称
            dataset_subset: 数据集子集
            tasks: 任务列表
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.device = device
        
        # 获取数据集处理器
        self.dataset_handler = get_dataset_handler(dataset_name, dataset_subset, tasks)
        
        # 创建 BM25 检索器 (用于构建初始示例池)
        self.bm25_retriever = BM25Retriever(
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            pool_size=config.global_pool_size,
            use_question_only=config.bm25_use_question_only,
            k1=config.bm25_k1,
            b=config.bm25_b,
            seed=config.seed,
            cache_dir=os.path.join(config.output_dir, "cache"),
            tasks=tasks
        )
        
        # 创建 KV Pool Manager
        self.kv_pool = KVPoolManager(
            model=model,
            tokenizer=tokenizer,
            dataset_handler=self.dataset_handler,
            pool_size=config.global_pool_size,
            cache_dir=os.path.join(config.output_dir, "kv_pool"),
            device=device,
            mode=config.mode  # 传递模式参数
        )
        
        # 创建 Query Encoder
        self.query_encoder = QueryEncoder(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # 创建 Shot Ranker
        self.shot_ranker = ShotRanker(
            kv_pool_manager=self.kv_pool,
            verbose=config.verbose
        )
        
        # 创建 Probe Selector
        self.probe_selector = ProbeSelector(
            model=model,
            tokenizer=tokenizer,
            kv_pool_manager=self.kv_pool,
            window_size=config.window_size,
            entropy_threshold=config.entropy_threshold,
            max_rounds=config.max_probe_rounds,
            device=device,
            verbose=config.verbose,
            mode=config.mode,  # 传递模式
            paper_num_questions=config.paper_num_questions  # 传递 paper 参数
        )
        
        # 创建 KV Assembler
        self.kv_assembler = KVAssembler(
            model=model,
            tokenizer=tokenizer,
            kv_pool_manager=self.kv_pool,
            device=device,
            mode=config.mode,  # 传递模式
            paper_num_questions=config.paper_num_questions  # 传递 paper 参数
        )
        
        logging.info(f"ManyShotKVEvaluator 初始化完成: dataset={dataset_name}/{dataset_subset}")
    
    def _get_system_prompt(self) -> str:
        """根据数据集类型和模式获取 system prompt"""
        name_l = (self.dataset_name or "").lower()
        
        # CoT-Collection 数据集（医疗、NLI 等多种任务）
        if "cot-collection" in name_l or "modelscope" in name_l:
            if self.config.mode == "io":
                return (
                    "You are a helpful assistant. "
                    "Provide the final answer directly and clearly."
                )
            else:  # cot or paper
                return (
                    "You are a helpful assistant. "
                    "Carefully analyze the given task and provide your reasoning step by step. "
                    "Then provide the final answer clearly at the end."
                )
        
        if "aqua" in name_l:
            if self.config.mode == "io":
                return (
                    "You are a precise math assistant. "
                    "Answer multiple choice questions by providing the correct letter (A, B, C, D, or E). "
                    "End your response with 'The answer is X' where X is the correct option letter."
                )
            else:
                return (
                    "You are a precise math assistant. "
                    "Answer multiple choice questions by analyzing the options and providing the correct letter (A, B, C, D, or E). "
                    "End your response with 'The answer is X' where X is the correct option letter."
                )
        
        if "physics" in name_l:
            if self.config.mode == "io":
                return (
                    "You are a physics expert. "
                    "Provide the final numerical answer clearly."
                )
            else:
                return (
                    "You are a physics expert. "
                    "Solve the given physics problems step by step. "
                    "Provide the final numerical answer clearly at the end."
                )
        
        if "competition_math" in name_l or "math" in name_l:
            if self.config.mode == "io":
                return (
                    "You are an expert mathematician. "
                    "Provide the final answer clearly. "
                    "If applicable, use boxed{} to highlight the final answer."
                )
            else:
                return (
                    "You are an expert mathematician. "
                    "Solve the given mathematical problems step by step. "
                    "Show your work clearly and provide the final answer. "
                    "If applicable, use boxed{} to highlight the final answer."
                )
        
        if "svamp" in name_l:
            if self.config.mode == "io":
                return (
                    "You are a helpful math assistant. "
                    "Provide only the numerical answer."
                )
            else:
                return (
                    "You are a helpful math assistant. "
                    "Solve the given word problem step by step. "
                    "Provide only the numerical answer at the end."
                )
        
        # 默认 GSM8K 风格 - 强调正确的答案格式
        if self.config.mode == "io":
            return (
                "You are a precise math assistant. "
                "Provide the final answer in the format '#### <number>'. "
                "IMPORTANT: The answer must be written as '####' followed by a space and then the number. "
                "For example: '#### 42' or '#### 3.14'. "
                "Do NOT write the number before ####."
            )
        else:
            return (
                "You are a precise math assistant. "
                "Solve the given math problem step by step. "
                "IMPORTANT: End your answer with '#### <number>' where <number> is the final numerical answer. "
                "The format must be '####' followed by a space and then the number. "
                "For example: '#### 42' or '#### 3.14'. "
                "Do NOT write the number before ####."
            )
    
    def evaluate(self, test_set, output_prefix: str) -> Dict:
        """
        运行完整评估
        
        Args:
            test_set: 测试集
            output_prefix: 输出文件前缀
        
        Returns:
            metrics: 评估指标
        """
        # 🔥 设置 probe_selector 的输出目录（与结果文件放在同一目录）
        output_dir = os.path.dirname(os.path.abspath(output_prefix))
        self.probe_selector.output_dir = output_dir
        self.probe_selector._probe_example_saved = False  # 重置标志位
        
        # 步骤 1: 构建 BM25 索引
        logging.info("=" * 80)
        logging.info("步骤 1/6: 构建 BM25 索引")
        logging.info("=" * 80)
        self.bm25_retriever.build_index()
        
        # 步骤 2: 离线 Prefilling - 构建 KV 池
        logging.info("=" * 80)
        logging.info("步骤 2/6: 离线 Prefilling (1024-shot KV Cache 池)")
        logging.info("=" * 80)
        pool_examples = self.bm25_retriever.pool_examples
        
        # ✅ 构建 Paper 模式的配置
        paper_config = None
        if self.config.mode == "paper":
            # 使用前 4 个 shots 作为 fullshots（这里简单使用第 0-3 个）
            paper_config = {
                'fullshot_ids': list(range(4)),  # 可以根据需要调整
                'num_questions': self.config.paper_num_questions
            }
        
        # 获取 system prompt
        system_prompt = self._get_system_prompt()
        
        # ✅ 构建 KV 池，同时缓存固定部分
        self.kv_pool.build_kv_pool(
            pool_examples,
            system_prompt=system_prompt,
            paper_config=paper_config
        )
        
        # 保存 KV 池信息
        kv_pool_info = self.kv_pool.get_pool_info()
        with open(f"{output_prefix}_kv_pool_info.json", 'w', encoding='utf-8') as f:
            json.dump(kv_pool_info, f, ensure_ascii=False, indent=2)
        
        # 准备输出文件
        jsonl_path = f"{output_prefix}.jsonl"
        meta_path = f"{output_prefix}_metrics.json"
        txt_path = f"{output_prefix}_report.txt"
        probe_details_path = f"{output_prefix}_probe_details.jsonl"
        
        # 步骤 3-6: 在线评估循环
        logging.info("=" * 80)
        logging.info("步骤 3-6: 在线评估 (Query编码 -> Shot排序 -> 探针选择 -> KV拼装 -> 生成)")
        logging.info("=" * 80)
        
        n_eval = len(test_set)
        prompt_text = ""  # ✅ Prompt 已经在固定部分了，不需要重复
        
        # Tokenize prompt (空文本）
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").input_ids.squeeze(0)
        
        # 统计指标
        total_em_str = 0.0
        total_contains = 0.0
        total_token_f1 = 0.0
        n_numeric_ok = 0
        total_relerr = []
        total_latency = []
        total_in_tok = []
        total_out_tok = []
        num_shots_list = []
        
        iterator = tqdm(range(n_eval), total=n_eval, desc="Many-Shot KV 评估")
        
        with open(jsonl_path, "w", encoding="utf-8") as fout, \
             open(probe_details_path, "w", encoding="utf-8") as fprobe:
            
            for i in iterator:
                ex = test_set[i]
                q, ref = self.dataset_handler.format_example_cot(ex)
                gold_final = self.dataset_handler.extract_gold_answer(ex)
                
                # Tokenize query
                query_tokens = self.tokenizer(q, return_tensors="pt").input_ids.squeeze(0)
                
                t_start = time.time()
                
                # 步骤 3: Query 编码
                query_repr = self.query_encoder.encode_query(prompt_tokens, query_tokens)
                
                # 步骤 4: Shot 排序
                ranked_shots = self.shot_ranker.rank_shots(query_repr)
                
                # 步骤 5: 探针选择
                selected_shots, probe_history = self.probe_selector.select_shots_with_probe(
                    ranked_shots, prompt_tokens, query_tokens, q  # 传递 query 文本
                )
                
                num_shots = len(selected_shots)
                num_shots_list.append(num_shots)
                
                # 保存探针详情
                fprobe.write(json.dumps({
                    "question_idx": i,
                    "question": q[:200],
                    "num_selected_shots": num_shots,
                    "probe_history": probe_history
                }, ensure_ascii=False) + "\n")
                
                # 步骤 6: KV 拼装与生成
                response, gen_info = self.kv_assembler.generate_with_kv_cache(
                    selected_shots, prompt_tokens, query_tokens,
                    max_new_tokens=self.config.gen_tokens,
                    query_text=q  # 传递 query 文本
                )
                
                t_end = time.time()
                latency = t_end - t_start
                
                # 提取预测答案（含稳健后处理）
                pred_final = self.dataset_handler.extract_prediction(response)
                if not pred_final or not any(ch.isdigit() for ch in pred_final):
                    import re
                    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
                    if m:
                        pred_final = m.group(1)
                    else:
                        m2 = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
                        if m2:
                            pred_final = m2.group(1)
                        else:
                            nums = re.findall(r"-?\d+(?:\.\d+)?", response)
                            if nums:
                                pred_final = nums[-1]
                
                # 评估
                compare_results = evaluate_answer(pred_final, gold_final)
                numeric_ok, relerr, em_str, contains, tf1 = (
                    compare_results["numeric_ok"],
                    compare_results["relerr"],
                    compare_results["em_str"],
                    compare_results["contains"],
                    compare_results["tf1"]
                )
                
                # 记录
                record = {
                    "question": q,
                    "reference_answer": ref,
                    "gold_answer": gold_final,
                    "model_output": response,
                    "pred_answer": pred_final,
                    "num_shots": num_shots,
                    "selected_shot_ids": selected_shots,
                    "EM_string": em_str,
                    "Contains": contains,
                    "Token_F1": tf1,
                    "Numeric_EM": numeric_ok,
                    "rel_error": relerr,
                    "time_spent": latency,
                    "kv_tokens": gen_info['total_kv_tokens'],
                    "output_tokens": gen_info['output_tokens']
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # 累计
                total_em_str += em_str
                total_contains += contains
                total_token_f1 += tf1
                n_numeric_ok += numeric_ok
                if relerr is not None:
                    total_relerr.append(relerr)
                total_latency.append(latency)
                total_in_tok.append(gen_info['total_kv_tokens'])
                total_out_tok.append(gen_info['output_tokens'])
        
        # 计算汇总指标
        logging.info("=" * 80)
        logging.info("汇总评估结果")
        logging.info("=" * 80)
        
        count = n_eval
        relerr_mean = float(np.mean(total_relerr)) if total_relerr else 0.0
        relerr_median = float(np.median(total_relerr)) if total_relerr else 0.0
        
        acc_numeric = n_numeric_ok / count if count else 0.0
        
        meta = {
            "dataset": f"{self.dataset_name}/{self.dataset_subset}",
            "count": count,
            "mode": f"manyshot_kv_cache_{self.config.mode}",  # 添加模式信息
            "global_pool_size": self.config.global_pool_size,
            "window_size": self.config.window_size,
            "entropy_threshold": self.config.entropy_threshold,
            "num_shots_mean": float(np.mean(num_shots_list)),
            "num_shots_median": float(np.median(num_shots_list)),
            "num_shots_min": int(np.min(num_shots_list)),
            "num_shots_max": int(np.max(num_shots_list)),
            "acc_numeric": acc_numeric,
            "acc_numeric_n": n_numeric_ok,
            "em_string": total_em_str / count if count else 0.0,
            "contains": total_contains / count if count else 0.0,
            "token_f1": total_token_f1 / count if count else 0.0,
            "relerr_mean": relerr_mean,
            "relerr_median": relerr_median,
            "avg_latency_s": float(np.mean(total_latency)) if total_latency else 0.0,
            "avg_kv_tokens": float(np.mean(total_in_tok)) if total_in_tok else 0.0,
            "avg_out_tokens": float(np.mean(total_out_tok)) if total_out_tok else 0.0,
        }
        
        # 保存指标
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 保存文本报告
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Many-Shot KV Cache 评估报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"数据集: {meta['dataset']}\n")
            f.write(f"测试样本数: {count}\n")
            f.write(f"全局示例池: {meta['global_pool_size']}\n")
            f.write(f"窗口大小: {meta['window_size']}\n")
            f.write(f"熵阈值: {meta['entropy_threshold']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Shot 数量统计\n")
            f.write("=" * 80 + "\n")
            f.write(f"平均: {meta['num_shots_mean']:.2f}\n")
            f.write(f"中位数: {meta['num_shots_median']:.1f}\n")
            f.write(f"范围: [{meta['num_shots_min']}, {meta['num_shots_max']}]\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("准确率指标\n")
            f.write("=" * 80 + "\n")
            f.write(f"Acc (Numeric-EM): {meta['acc_numeric']:.4f} ({meta['acc_numeric_n']}/{count})\n")
            f.write(f"EM (String): {meta['em_string']:.4f}\n")
            f.write(f"Contains: {meta['contains']:.4f}\n")
            f.write(f"Token-F1: {meta['token_f1']:.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("性能指标\n")
            f.write("=" * 80 + "\n")
            f.write(f"平均延迟: {meta['avg_latency_s']:.3f} s\n")
            f.write(f"平均 KV tokens: {meta['avg_kv_tokens']:.1f}\n")
            f.write(f"平均输出 tokens: {meta['avg_out_tokens']:.1f}\n")
        
        logging.info(f"✓ 评估完成！准确率: {meta['acc_numeric']:.4f}, 平均shots: {meta['num_shots_mean']:.2f}")
        
        return meta
