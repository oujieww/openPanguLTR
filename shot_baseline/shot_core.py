"""
Shot Baseline 核心评估模块
支持三种模式：CoT (Chain-of-Thought), IO (Input-Output), Paper
"""
import os
import sys
import json
import time
import logging
import random
import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加根路径以定位 util 包
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# 从 util 导入评估函数
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


class ShotEvaluator:
    """Shot Baseline 评估器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        config,
        dataset_handler,
        dataset_name: str,
        dataset_subset: str = None,
        train_pool: List = None
    ):
        """
        初始化评估器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            config: Shot 配置
            dataset_handler: 数据集处理器
            dataset_name: 数据集名称
            dataset_subset: 数据集子集
            train_pool: 训练集示例池（用于随机选择 shots）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_handler = dataset_handler
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.train_pool = train_pool or []
        
        logging.info(f"ShotEvaluator 初始化完成: mode={config.mode}, dataset={dataset_name}/{dataset_subset}")
        logging.info(f"训练池大小: {len(self.train_pool)}")
    
    def _get_system_prompt(self) -> str:
        """根据数据集类型获取 system prompt"""
        name_l = (self.dataset_name or "").lower()
        
        # CoT-Collection 数据集（医疗、NLI 等多种任务）
        if "cot-collection" in name_l or "modelscope" in name_l:
            return (
                "You are a helpful assistant. "
                "Carefully analyze the given task and provide your reasoning step by step. "
                "Then provide the final answer clearly at the end."
            )
        
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
    
    def _select_random_shots(self, test_question: str, num_shots: int) -> List:
        """
        随机选择 shots，确保与测试问题不重复
        
        Args:
            test_question: 测试问题
            num_shots: 需要的 shot 数量
        
        Returns:
            选中的示例列表
        """
        if not self.train_pool or num_shots <= 0:
            return []
        
        # 过滤掉与测试问题相同的示例
        test_q_norm = test_question.strip().lower()
        available_pool = []
        for ex in self.train_pool:
            q, _ = self.dataset_handler.format_example_cot(ex)
            if q.strip().lower() != test_q_norm:
                available_pool.append(ex)
        
        if not available_pool:
            logging.warning(f"没有可用的示例（训练池大小: {len(self.train_pool)}）")
            return []
        
        # 随机选择
        actual_num = min(num_shots, len(available_pool))
        selected = random.sample(available_pool, actual_num)
        
        return selected
    
    def _build_prompt_cot(self, test_question: str, num_shots: int) -> Tuple[str, int]:
        """
        构建 CoT 模式的 prompt
        每个 shot 包含：问题 + 解答过程 + 答案
        
        Returns:
            (prompt_text, actual_num_shots)
        """
        shots = self._select_random_shots(test_question, num_shots)
        
        examples_text = []
        for ex in shots:
            q, cot = self.dataset_handler.format_example_cot(ex)
            examples_text.append(f"Problem: {q}\nSolution: {cot}")
        
        if examples_text:
            prompt = "\n\n".join(examples_text) + f"\n\nProblem: {test_question}\nSolution:"
        else:
            prompt = f"Problem: {test_question}\nSolution:"
        
        return prompt, len(shots)
    
    def _build_prompt_io(self, test_question: str, num_shots: int) -> Tuple[str, int]:
        """
        构建 IO 模式的 prompt
        每个 shot 包含：问题 + 答案（无解答过程）
        
        Returns:
            (prompt_text, actual_num_shots)
        """
        shots = self._select_random_shots(test_question, num_shots)
        
        examples_text = []
        for ex in shots:
            q, _ = self.dataset_handler.format_example_cot(ex)
            answer = self.dataset_handler.extract_gold_answer(ex)
            examples_text.append(f"Problem: {q}\nAnswer: {answer}")
        
        if examples_text:
            prompt = "\n\n".join(examples_text) + f"\n\nProblem: {test_question}\nAnswer:"
        else:
            prompt = f"Problem: {test_question}\nAnswer:"
        
        return prompt, len(shots)
    
    def _build_prompt_paper(self, test_question: str, k_full: int, num_questions: int) -> Tuple[str, int, int]:
        """
        构建 Paper 模式的 prompt
        按照 AdaCache 格式：
        1. 先展示 question-only shots（类似示例）
        2. 添加分隔符和提示
        3. 再展示 fullshots（完整示例）
        4. 最后是测试问题
        
        Returns:
            (prompt_text, actual_k_full, actual_num_questions)
        """
        # 选择 fullshots（问题 + 过程 + 答案）
        fullshots = self._select_random_shots(test_question, k_full)
        
        # 选择 question-only shots（只有问题）
        # 需要确保不与 fullshots 和测试问题重复
        used_questions = set()
        for ex in fullshots:
            q, _ = self.dataset_handler.format_example_cot(ex)
            used_questions.add(q.strip().lower())
        used_questions.add(test_question.strip().lower())
        
        question_only_pool = []
        for ex in self.train_pool:
            q, _ = self.dataset_handler.format_example_cot(ex)
            if q.strip().lower() not in used_questions:
                question_only_pool.append(ex)
        
        actual_num_q = min(num_questions, len(question_only_pool))
        question_shots = random.sample(question_only_pool, actual_num_q) if question_only_pool else []
        
        # 构建 prompt（按照 AdaCache 的顺序）
        parts = []
        
        # 1. 先添加引导语（如果有 question-only shots）
        if question_shots:
            parts.append("You will be provided Problems similar to the ones below:")
            for ex in question_shots:
                q, _ = self.dataset_handler.format_example_cot(ex)
                parts.append(f"Problem: {q}")
        
        # 2. 添加分隔符和提示
        if fullshots:
            if question_shots:
                parts.append("—")  # 分隔符
            parts.append("Now, I am going to give you a series of demonstrations of Problems and Solutions to specify the output format.")
            parts.append("When you respond, think step by step, but your last line must be exactly of the form '#### <final_answer>'.")
            
            # 3. 添加 fullshots
            for ex in fullshots:
                q, cot = self.dataset_handler.format_example_cot(ex)
                parts.append(f"Problem: {q}\nSolution: {cot}")
            
            # 添加最后的分隔符
            parts.append("—")
        
        # 4. 添加测试问题
        if parts:
            prompt = "\n".join(parts) + f"\nProblem: {test_question}\nSolution:"
        else:
            prompt = f"Problem: {test_question}\nSolution:"
        
        return prompt, len(fullshots), len(question_shots)
    
    def _save_pool_info(self, pool_info_path: str):
        """保存示例池信息到 JSON 文件"""
        pool_info = {
            "status": "ready",
            "pool_size": len(self.train_pool),
            "dataset": f"{self.dataset_name}/{self.dataset_subset or 'default'}",
            "mode": self.config.mode,
            "selection_method": "random",
            "seed": self.config.seed,
        }
        
        # 添加模式特定信息
        if self.config.mode == "cot":
            pool_info["num_shots"] = self.config.num_shots
            pool_info["shot_type"] = "fullshot (question + reasoning + answer)"
        elif self.config.mode == "io":
            pool_info["num_shots"] = self.config.num_shots
            pool_info["shot_type"] = "shortshot (question + answer only)"
        else:  # paper
            pool_info["paper_k_full"] = self.config.paper_k_full
            pool_info["paper_num_questions"] = self.config.paper_num_questions
            pool_info["shot_type"] = "mixed (fullshots + question-only shots)"
        
        # 添加示例预览（前5个）
        examples_preview = []
        preview_count = min(5, len(self.train_pool))
        
        for idx in range(preview_count):
            try:
                ex = self.train_pool[idx]
                q, cot = self.dataset_handler.format_example_cot(ex)
                answer = self.dataset_handler.extract_gold_answer(ex)
                
                # 截断长文本
                q_preview = q[:150] + "..." if len(q) > 150 else q
                cot_preview = cot[:150] + "..." if len(cot) > 150 else cot
                
                examples_preview.append({
                    "index": idx,
                    "question": q_preview,
                    "answer_preview": cot_preview if self.config.mode == "cot" else answer
                })
            except Exception as e:
                logging.warning(f"处理示例 {idx} 失败: {e}")
                continue
        
        pool_info["examples_preview"] = examples_preview
        
        # 保存到文件
        try:
            with open(pool_info_path, "w", encoding="utf-8") as f:
                json.dump(pool_info, f, ensure_ascii=False, indent=2)
            logging.info(f"示例池信息已保存: {pool_info_path}")
        except Exception as e:
            logging.warning(f"保存示例池信息失败: {e}")
    
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
    
    def _truncate_response(self, response: str) -> str:
        """
        截断模型输出，只保留第一个完整答案
        
        问题：模型可能在回答完第一个问题后继续生成其他内容
        解决：在各种结束标记处截断
        """
        import re
        
        # 方法 1: 在 "#### " + 答案 后截断
        # 匹配模式：#### 答案 后面可能跟着换行符和新问题
        match = re.search(r'(####\s*[\-]?\d+(?:\.\d+)?)', response)
        if match:
            end_pos = match.end()
            # 检查后面是否有新问题开始
            remaining = response[end_pos:]
            # 如果后面有 "Problem:" 或重复内容，则截断
            if 'Problem:' in remaining or 'problem:' in remaining.lower():
                response = response[:end_pos].strip()
                return response
        
        # 方法 2: 在 "Problem:" 重复出现时截断
        # 查找第二个 "Problem:" 的位置
        first_problem = response.find('Problem:')
        if first_problem != -1:
            second_problem = response.find('Problem:', first_problem + 1)
            if second_problem != -1:
                response = response[:second_problem].strip()
                return response
        
        # 方法 3: 在双换行 + 新内容开始时截断
        # 查找 "\n\n" 后面跟着新问题
        parts = response.split('\n\n')
        if len(parts) > 1:
            result_parts = [parts[0]]
            for part in parts[1:]:
                part_lower = part.strip().lower()
                if part_lower.startswith('problem:') or part_lower.startswith('solution:'):
                    break
                result_parts.append(part)
            response = '\n\n'.join(result_parts)
        
        return response.strip()
    
    def evaluate(self, test_set, output_prefix: str) -> Dict:
        """
        运行评估
        
        Args:
            test_set: 测试集
            output_prefix: 输出文件前缀
        
        Returns:
            metrics: 评估指标
        """
        # 准备输出文件
        jsonl_path = f"{output_prefix}.jsonl"
        meta_path = f"{output_prefix}_metrics.json"
        txt_path = f"{output_prefix}_report.txt"
        prompt_example_path = f"{output_prefix}_prompt_example.txt"
        pool_info_path = f"{output_prefix}_pool_info.json"
        
        # 保存示例池信息
        self._save_pool_info(pool_info_path)
        
        # 评估循环
        logging.info("=" * 60)
        logging.info(f"开始 {self.config.mode.upper()} 模式评估")
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
        total_shots_used = []
        
        # 用于保存第一个 prompt 示例
        first_prompt_saved = False
        
        iterator = tqdm(range(n_eval), total=n_eval, desc=f"{self.config.mode.upper()} 评估")
        
        with open(jsonl_path, "w", encoding="utf-8") as fout, \
             torch.inference_mode():
            
            for i in iterator:
                ex = test_set[i]
                q, ref = self.dataset_handler.format_example_cot(ex)
                gold_final = self.dataset_handler.extract_gold_answer(ex)
                
                # 根据模式构建 prompt
                if self.config.mode == "cot":
                    user_content, num_shots = self._build_prompt_cot(q, self.config.num_shots)
                    mode_info = f"CoT-{num_shots}shot"
                elif self.config.mode == "io":
                    user_content, num_shots = self._build_prompt_io(q, self.config.num_shots)
                    mode_info = f"IO-{num_shots}shot"
                else:  # paper
                    user_content, k_full, num_q = self._build_prompt_paper(
                        q, self.config.paper_k_full, self.config.paper_num_questions
                    )
                    num_shots = k_full + num_q
                    mode_info = f"Paper-{k_full}full+{num_q}q"
                
                total_shots_used.append(num_shots)
                
                # 构建消息
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                
                try:
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = f"{system_prompt}\n\n{user_content}\n"
                
                # 保存第一个 prompt 示例
                if not first_prompt_saved:
                    try:
                        with open(prompt_example_path, "w", encoding="utf-8") as f_prompt:
                            f_prompt.write("=" * 80 + "\n")
                            f_prompt.write(f"Shot Baseline ({self.config.mode.upper()}) Prompt 示例\n")
                            f_prompt.write("=" * 80 + "\n\n")
                            f_prompt.write(f"测试问题索引: {i}\n")
                            f_prompt.write(f"问题: {q[:200]}...\n")
                            f_prompt.write(f"模式: {mode_info}\n")
                            f_prompt.write(f"Shot 数: {num_shots}\n")
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
                            f_prompt.write(f"模式: {self.config.mode}\n")
                            if self.config.mode == "paper":
                                f_prompt.write(f"Fullshot 数: {self.config.paper_k_full}\n")
                                f_prompt.write(f"Question-only 数: {self.config.paper_num_questions}\n")
                            else:
                                f_prompt.write(f"Shot 数: {self.config.num_shots}\n")
                            f_prompt.write(f"评测样本数: {self.config.eval_samples}\n")
                            f_prompt.write(f"生成 tokens: {self.config.gen_tokens}\n")
                            f_prompt.write(f"随机种子: {self.config.seed}\n")
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
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                
                # 尝试获取其他 eos token
                if hasattr(self.tokenizer, "convert_tokens_to_ids"):
                     # Qwen 系列可能使用 <|im_end|>, <|endoftext|> 等
                     extra_eos = ["<|im_end|>", "<|endoftext|>", "</s>"]
                     for t in extra_eos:
                         tid = self.tokenizer.convert_tokens_to_ids(t)
                         if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                             if eos_ids is None:
                                 eos_ids = [tid]
                             elif tid not in eos_ids:
                                 eos_ids.append(tid)
                                 
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
                
                # 🔥 截断输出：只保留第一个完整答案
                response = self._truncate_response(response)
                
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
                    "mode": mode_info,
                    "num_shots": num_shots,
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
        logging.info("汇总评估结果")
        logging.info("=" * 60)
        
        count = n_eval
        avg_shots = float(np.mean(total_shots_used)) if total_shots_used else 0.0
        relerr_mean = float(np.mean(total_relerr)) if total_relerr else 0.0
        relerr_median = float(np.median(total_relerr)) if total_relerr else 0.0
        
        acc_numeric = n_numeric_ok / count if count else 0.0
        # 容差准确率
        if total_relerr:
            acc_tol_1e4 = np.mean([1.0 if (e is not None and e <= 1e-4) else 0.0 for e in total_relerr])
            acc_tol_1e3 = np.mean([1.0 if (e is not None and e <= 1e-3) else 0.0 for e in total_relerr])
            acc_tol_1e2 = np.mean([1.0 if (e is not None and e <= 1e-2) else 0.0 for e in total_relerr])
        else:
            acc_tol_1e4 = 0.0
            acc_tol_1e3 = 0.0
            acc_tol_1e2 = 0.0
        
        em_string = total_em_str / count if count else 0.0
        contains = total_contains / count if count else 0.0
        token_f1 = total_token_f1 / count if count else 0.0
        
        meta = {
            "dataset": f"{self.dataset_name}/{self.dataset_subset or 'default'}",
            "mode": self.config.mode,
            "avg_num_shots": avg_shots,
            "count": count,
            "acc_numeric": acc_numeric,
            "acc_numeric_n": n_numeric_ok,
            "em_string": em_string,
            "contains": contains,
            "token_f1": token_f1,
            "acc_tol_1e-4": acc_tol_1e4,
            "acc_tol_1e-3": acc_tol_1e3,
            "acc_tol_1e-2": acc_tol_1e2,
            "relerr_mean": relerr_mean,
            "relerr_median": relerr_median,
            "total_time_s": float(np.sum(total_latency)) if total_latency else float("nan"),
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
            f.write(f"Shot Baseline ({self.config.mode.upper()}) 评估报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据集: {meta['dataset']}\n")
            f.write(f"测试样本数: {count}\n")
            f.write(f"模式: {self.config.mode}\n")
            f.write(f"平均 Shot 数: {avg_shots:.1f}\n\n")
            
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
            f.write(f"总耗时 (s): {meta['total_time_s']:.3f}\n")
            f.write(f"平均延迟 (s): {meta['avg_latency_s']:.3f}\n")
            f.write(f"平均输入/输出 tokens: {meta['avg_in_tokens']:.1f}/{meta['avg_out_tokens']:.1f}\n")
            f.write(f"平均 tokens/s: {meta['avg_tokpersec']:.2f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("输出文件\n")
            f.write("=" * 60 + "\n")
            f.write(f"JSONL: {jsonl_path}\n")
            f.write(f"Metrics: {meta_path}\n")
            f.write(f"Pool Info: {pool_info_path}\n")
            f.write(f"Prompt Example: {prompt_example_path}\n")
        
        logging.info(f"✓ 评估完成！结果已保存到 {output_prefix}*")
        logging.info(f"  - 准确率 (Numeric-EM): {meta['acc_numeric']:.4f}")
        logging.info(f"  - 平均 Shot 数: {avg_shots:.1f}")
        
        return meta
