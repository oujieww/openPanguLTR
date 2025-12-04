"""
AdaCache 核心逻辑
整合 BM25 检索和探针机制，实现自适应示例选择
"""
import sys
import os
import json
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

# 添加 util 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'util'))

from util.dataset_handlers import get_dataset_handler

# 从 util 导入 new_metrics 中的 evaluate_answer
try:
    from util.new_metrics import evaluate_answer
except ImportError:
    # 如果 new_metrics 不存在，使用本地定义的版本
    from util.metrics_utils import (
        normalize_answer,
        token_f1_pair,
        parse_number_from_text,
        numeric_equal,
        relative_error
    )
    
    def evaluate_answer(pred_final: str, gold_final: str) -> dict:
        """评估答案（备用版本）"""
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
from probe_mechanism import ProbeMechanism
from config import AdaCacheConfig


class AdaCacheEvaluator:
    """AdaCache 评估器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        config: AdaCacheConfig,
        dataset_name: str,
        dataset_subset: str = None,
        tasks: str = None
    ):
        """
        初始化 AdaCache 评估器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            config: AdaCache 配置
            dataset_name: 数据集名称
            dataset_subset: 数据集子集
            tasks: 任务列表（逗号分隔）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        
        # 获取数据集处理器
        self.dataset_handler = get_dataset_handler(dataset_name, dataset_subset, tasks)
        
        # 创建 BM25 检索器
        self.retriever = BM25Retriever(
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
        
        # 创建探针机制
        self.probe_mechanism = ProbeMechanism(
            model=model,
            tokenizer=tokenizer,
            probe_question=config.probe_question,
            entropy_threshold=config.entropy_threshold,
            window_size=config.window_size,
            max_rounds=config.max_probe_rounds,
            dataset_handler=self.dataset_handler,
            paper_k_full=config.paper_k_full,
            verbose=config.verbose
        )
        
        logging.info(f"AdaCacheEvaluator 初始化完成: dataset={dataset_name}/{dataset_subset}")
    
    def _build_paper_prompt(
        self,
        problems_only_examples: List[Dict],
        solved_examples: List[Dict],
        final_question: str
    ) -> str:
        """
        构建 paper 格式的 prompt（用于最终推理）
        
        Args:
            problems_only_examples: 只列题目的示例
            solved_examples: 完整示例（问题-过程-答案）
            final_question: 最终问题
        
        Returns:
            user_content: 用户内容
        """
        lines = []
        
        # 第一部分：只列题目
        if len(problems_only_examples) > 0:
            lines.append("You will be provided Problems similar to the ones below:")
            for ex in problems_only_examples:
                q, _ = self.dataset_handler.format_example_cot(ex)
                lines.append(f"Problem: {q}")
            lines.append("—")
        
        # 第二部分：完整示例
        if len(solved_examples) > 0:
            lines.append("Now, I am going to give you a series of demonstrations of Problems and Solutions to specify the output format.")
            lines.append("When you respond, think step by step, but your last line must be exactly of the form '#### <final_answer>'.")
            for ex in solved_examples:
                q, a_raw = self.dataset_handler.format_example_cot(ex)
                gold = self.dataset_handler.extract_gold_answer(ex)
                lines.append(f"Problem: {q}")
                lines.append("Solution:")
                lines.append(a_raw.rstrip())
                lines.append(f"Answer: {gold}.")
                lines.append(f"Final Answer: The final answer is {gold}.")
                lines.append("—")
        
        # 第三部分：最终问题
        lines.append(f"Problem: {final_question}")
        lines.append("Solution:")
        
        return "\n".join(lines)
    
    def _get_system_prompt(self) -> str:
        """
        根据数据集类型获取 system prompt
        参考 baseline/eval_core.py 的实现
        """
        name_l = (self.dataset_name or "").lower()
        
        if "aqua" in name_l:
            return (
                "You are a precise math assistant. "
                "Answer multiple choice questions by analyzing the options and providing the correct letter (A, B, C, D, or E). "
                "End your response with 'The answer is X' where X is the correct option letter."
            )
        
        if "physics" in name_l:
            return (
                "You are a physics expert. "
                "Solve the given physics problems step by step. "
                "Provide the final numerical answer clearly at the end."
            )
        
        if "competition_math" in name_l or "math" in name_l:
            return (
                "You are an expert mathematician. "
                "Solve the given mathematical problems step by step. "
                "Show your work clearly and provide the final answer. "
                "If applicable, use boxed{} to highlight the final answer."
            )
        
        if "svamp" in name_l:
            return (
                "You are a helpful math assistant. "
                "Solve the given word problem step by step. "
                "Provide only the numerical answer at the end."
            )
        
        # 默认 GSM8K 风格 - 强调正确的答案格式
        return (
            "You are a precise math assistant. "
            "Solve the given math problem step by step. "
            "IMPORTANT: End your answer with '#### <number>' where <number> is the final numerical answer. "
            "The format must be '####' followed by a space and then the number. "
            "For example: '#### 42' or '#### 3.14'. "
            "Do NOT write the number before ####."
        )
    
    def _pick_input_device(self):
        """选择输入设备"""
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            device_map = self.model.hf_device_map
            first_layer = sorted(device_map.keys())[0]
            return device_map[first_layer]
        elif hasattr(self.model, 'device'):
            return self.model.device
        else:
            return torch.device('cpu')
    
    def evaluate(self, test_set, output_prefix: str) -> Dict:
        """
        运行 AdaCache 评估
        
        Args:
            test_set: 测试集
            output_prefix: 输出文件前缀
        
        Returns:
            metrics: 评估指标
        """
        # 构建 BM25 索引
        logging.info("=" * 60)
        logging.info("步骤 1/3: 构建 BM25 索引")
        logging.info("=" * 60)
        self.retriever.build_index()
        
        # 保存示例池信息
        pool_info_path = f"{output_prefix}_pool_info.json"
        self.retriever.save_pool_info(pool_info_path)
        
        # 准备输出文件
        jsonl_path = f"{output_prefix}.jsonl"
        meta_path = f"{output_prefix}_metrics.json"
        txt_path = f"{output_prefix}_report.txt"
        probe_details_path = f"{output_prefix}_probe_details.jsonl"
        prompt_example_path = f"{output_prefix}_prompt_example.txt"  # 新增：保存 prompt 示例
        
        # 评估循环
        logging.info("=" * 60)
        logging.info("步骤 2/3: 自适应示例选择与推理")
        logging.info("=" * 60)
        
        n_eval = len(test_set)
        system_prompt = self._get_system_prompt()
        
        # 统计指标
        total_em_str = 0.0
        total_contains = 0.0
        total_token_f1 = 0.0
        n_numeric_ok = 0
        total_relerr = []
        total_latency = []
        total_in_tok = []
        total_out_tok = []
        total_tps = []
        optimal_k_list = []
        
        # 用于保存第一个 prompt 示例
        first_prompt_saved = False
        
        iterator = tqdm(range(n_eval), total=n_eval, desc="AdaCache 评估")
        
        with open(jsonl_path, "w", encoding="utf-8") as fout, \
             open(probe_details_path, "w", encoding="utf-8") as fprobe, \
             torch.inference_mode():
            
            for i in iterator:
                ex = test_set[i]
                q, ref = self.dataset_handler.format_example_cot(ex)
                gold_final = self.dataset_handler.extract_gold_answer(ex)
                
                # 步骤 1: BM25 检索
                sorted_examples, scores = self.retriever.retrieve(q, top_k=self.config.max_examples)
                
                # 步骤 2: 探针机制确定最优示例数
                optimal_k, probe_history = self.probe_mechanism.determine_optimal_examples(
                    query_question=q,
                    sorted_examples=sorted_examples,
                    system_prompt=system_prompt
                )
                
                optimal_k_list.append(optimal_k)
                
                # 保存探针历史
                if self.config.save_probe_details:
                    fprobe.write(json.dumps({
                        "question_idx": i,
                        "question": q[:200],
                        "optimal_k": optimal_k,
                        "probe_history": probe_history
                    }, ensure_ascii=False) + "\n")
                
                # 步骤 3: 使用最优示例数构建 prompt 并推理
                selected_examples = sorted_examples[:optimal_k]
                
                # 分割示例：problems-only 和 solved
                k_full = min(self.config.paper_k_full, optimal_k)
                problems_only_count = max(0, optimal_k - k_full)
                
                problems_only = selected_examples[:problems_only_count]
                solved = selected_examples[problems_only_count:optimal_k]
                
                # 构建 prompt
                user_content = self._build_paper_prompt(problems_only, solved, q)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                
                try:
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = f"{system_prompt}\n\n{user_content}\n"
                
                # 保存第一个 prompt 示例（用于检查 prompt 模板）
                if not first_prompt_saved:
                    try:
                        with open(prompt_example_path, "w", encoding="utf-8") as f_prompt:
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write("AdaCache Prompt 示例\n")
                            f_prompt.write("=" * 80 + "\n\n")
                            f_prompt.write(f"测试问题索引: {i}\n")
                            f_prompt.write(f"问题: {q[:200]}...\n")
                            f_prompt.write(f"最优示例数: {optimal_k}\n")
                            f_prompt.write(f"Problems-only 示例数: {problems_only_count}\n")
                            f_prompt.write(f"完整示例数 (solved): {len(solved)}\n")
                            f_prompt.write("\n" + "=" * 80 + "\n")
                            f_prompt.write("System Prompt\n")
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write(system_prompt + "\n\n")
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write("User Content\n")
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write(user_content + "\n\n")
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write("完整 Prompt（应用 chat template 后）\n")
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write(text + "\n\n")
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write("配置信息\n")
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write(f"全局示例池大小: {self.config.global_pool_size}\n")
                            f_prompt.write(f"信息熵阈值: {self.config.entropy_threshold}\n")
                            f_prompt.write(f"窗口大小: {self.config.window_size}\n")
                            f_prompt.write(f"Paper K_full: {self.config.paper_k_full}\n")
                            f_prompt.write(f"最大示例数: {self.config.max_examples}\n")
                        first_prompt_saved = True
                        logging.info(f"Prompt 示例已保存: {prompt_example_path}")
                    except Exception as e:
                        logging.warning(f"保存 prompt 示例失败: {e}")
                
                # 编码
                model_inputs = self.tokenizer([text], return_tensors="pt")
                if getattr(self.tokenizer, "pad_token_id", None) is None:
                    self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
                
                # 移动到设备
                input_dev = self._pick_input_device()
                try:
                    model_inputs = model_inputs.to(input_dev)
                except Exception:
                    pass
                
                in_tok = int(model_inputs["input_ids"].shape[1])
                
                # 生成
                eos_ids = getattr(self.tokenizer, "eos_token_id", None)
                gen_kwargs = dict(
                    max_new_tokens=max(1, self.config.gen_tokens),
                    do_sample=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
                if eos_ids is not None:
                    gen_kwargs["eos_token_id"] = eos_ids
                
                t0 = time.time()
                outputs = self.model.generate(**model_inputs, **gen_kwargs)
                t1 = time.time()
                
                # 解码
                full_seq = outputs.sequences[0]
                prompt_len = model_inputs["input_ids"].shape[1]
                gen_ids = full_seq[prompt_len:]
                response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                out_tok = int(gen_ids.shape[0])
                latency = (t1 - t0)
                tps = (out_tok / latency) if latency > 0 else float("nan")
                
                # 提取预测答案
                pred_final = self.dataset_handler.extract_prediction(response)
                
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
                    "optimal_k": optimal_k,
                    "EM_string": em_str,
                    "Contains": contains,
                    "Token_F1": tf1,
                    "Numeric_EM": numeric_ok,
                    "rel_error": relerr,
                    "time_spent": latency,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "tokens_per_sec": tps
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
                total_in_tok.append(in_tok)
                total_out_tok.append(out_tok)
                total_tps.append(tps)
        
        # 计算汇总指标
        logging.info("=" * 60)
        logging.info("步骤 3/3: 汇总评估结果")
        logging.info("=" * 60)
        
        count = n_eval
        relerr_mean = float(np.mean(total_relerr)) if total_relerr else 0.0
        relerr_median = float(np.median(total_relerr)) if total_relerr else 0.0
        
        acc_numeric = n_numeric_ok / count if count else 0.0
        # 容差准确率：如果没有数据则返回 0.0
        if total_relerr:
            acc_tol_1e4 = np.mean([1.0 if (e is not None and e <= 1e-4) else 0.0 for e in total_relerr])
            acc_tol_1e3 = np.mean([1.0 if (e is not None and e <= 1e-3) else 0.0 for e in total_relerr])
            acc_tol_1e2 = np.mean([1.0 if (e is not None and e <= 1e-2) else 0.0 for e in total_relerr])
        else:
            acc_tol_1e4 = 0.0
            acc_tol_1e3 = 0.0
            acc_tol_1e2 = 0.0
        
        meta = {
            "dataset": f"{self.dataset_name}/{self.dataset_subset}",
            "count": count,
            "global_pool_size": self.config.global_pool_size,
            "entropy_threshold": self.config.entropy_threshold,
            "window_size": self.config.window_size,
            "paper_k_full": self.config.paper_k_full,
            "optimal_k_mean": float(np.mean(optimal_k_list)),
            "optimal_k_median": float(np.median(optimal_k_list)),
            "optimal_k_min": int(np.min(optimal_k_list)),
            "optimal_k_max": int(np.max(optimal_k_list)),
            "acc_numeric": acc_numeric,
            "acc_numeric_n": n_numeric_ok,
            "em_string": total_em_str / count if count else 0.0,
            "contains": total_contains / count if count else 0.0,
            "token_f1": total_token_f1 / count if count else 0.0,
            "acc_tol_1e-4": acc_tol_1e4,
            "acc_tol_1e-3": acc_tol_1e3,
            "acc_tol_1e-2": acc_tol_1e2,
            "relerr_mean": relerr_mean,
            "relerr_median": relerr_median,
            "avg_latency_s": float(np.mean(total_latency)) if total_latency else float("nan"),
            "avg_in_tokens": float(np.mean(total_in_tok)) if total_in_tok else float("nan"),
            "avg_out_tokens": float(np.mean(total_out_tok)) if total_out_tok else float("nan"),
            "avg_tokpersec": float(np.mean(total_tps)) if total_tps else float("nan"),
        }
        
        # 保存指标
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 保存文本报告
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("AdaCache 评估报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据集: {meta['dataset']}\n")
            f.write(f"测试样本数: {count}\n")
            f.write(f"全局示例池大小: {meta['global_pool_size']}\n")
            f.write(f"信息熵阈值: {meta['entropy_threshold']}\n")
            f.write(f"窗口大小: {meta['window_size']}\n")
            f.write(f"Paper K_full: {meta['paper_k_full']}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("自适应示例数统计\n")
            f.write("=" * 60 + "\n")
            f.write(f"平均示例数: {meta['optimal_k_mean']:.2f}\n")
            f.write(f"中位数示例数: {meta['optimal_k_median']:.1f}\n")
            f.write(f"最小示例数: {meta['optimal_k_min']}\n")
            f.write(f"最大示例数: {meta['optimal_k_max']}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("准确率指标\n")
            f.write("=" * 60 + "\n")
            f.write(f"Acc (Numeric-EM): {meta['acc_numeric']:.4f} ({meta['acc_numeric_n']}/{count})\n")
            f.write(f"EM (String): {meta['em_string']:.4f}\n")
            f.write(f"Contains: {meta['contains']:.4f}\n")
            f.write(f"Token-F1: {meta['token_f1']:.4f}\n")
            f.write(f"Acc@1e-4: {meta['acc_tol_1e-4']:.4f}, "
                   f"Acc@1e-3: {meta['acc_tol_1e-3']:.4f}, "
                   f"Acc@1e-2: {meta['acc_tol_1e-2']:.4f}\n")
            f.write(f"RelErr mean/median: {relerr_mean:.6f}/{relerr_median:.6f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("性能指标\n")
            f.write("=" * 60 + "\n")
            f.write(f"平均延迟 (s): {meta['avg_latency_s']:.3f}\n")
            f.write(f"平均输入/输出 tokens: {meta['avg_in_tokens']:.1f}/{meta['avg_out_tokens']:.1f}\n")
            f.write(f"平均 tokens/s: {meta['avg_tokpersec']:.2f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("输出文件\n")
            f.write("=" * 60 + "\n")
            f.write(f"JSONL: {jsonl_path}\n")
            f.write(f"Metrics: {meta_path}\n")
            f.write(f"Probe Details: {probe_details_path}\n")
            f.write(f"Pool Info: {pool_info_path}\n")
            f.write(f"Prompt Example: {prompt_example_path}\n")
        
        logging.info(f"✓ 评估完成！结果已保存到 {output_prefix}*")
        logging.info(f"  - 准确率 (Numeric-EM): {meta['acc_numeric']:.4f}")
        logging.info(f"  - 平均示例数: {meta['optimal_k_mean']:.2f}")
        
        return meta
