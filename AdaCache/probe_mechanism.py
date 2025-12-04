"""
探针机制
实现自适应示例分配的探针流程
"""
import sys
import os
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加根路径以定位 util 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from util.dataset_handlers import BaseDatasetHandler


class ProbeMechanism:
    """探针机制，用于动态确定最优示例数量"""
    
    def __init__(
        self,
        model,
        tokenizer,
        probe_question: str,
        entropy_threshold: float = 0.5,
        window_size: int = 4,
        max_rounds: int = 128,
        dataset_handler: BaseDatasetHandler = None,
        paper_k_full: int = 4,
        verbose: bool = True
    ):
        """
        初始化探针机制
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            probe_question: 探针问题文本
            entropy_threshold: 信息熵阈值 τ
            window_size: 每轮激活的示例数量 n
            max_rounds: 最大探针轮数
            dataset_handler: 数据集处理器
            paper_k_full: paper 模式下的固定完整示例数
            verbose: 是否输出详细日志
        """
        self.model = model
        self.tokenizer = tokenizer
        self.probe_question = probe_question
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.max_rounds = max_rounds
        self.dataset_handler = dataset_handler
        self.paper_k_full = paper_k_full
        self.verbose = verbose
        
        # 预编译 Yes/No token IDs
        self._prepare_yesno_tokens()
        
        logging.info(f"初始化 ProbeMechanism: threshold={entropy_threshold}, "
                    f"window_size={window_size}, max_rounds={max_rounds}")
    
    def _prepare_yesno_tokens(self):
        """预编译 Yes/No 的 token IDs（只支持英文）"""
        # 可能的 Yes/No 表达（英文）
        yes_variants = ["Yes", "yes", "YES"]
        no_variants = ["No", "no", "NO"]
        
        # 获取所有可能的 token IDs
        self.yes_token_ids = set()
        self.no_token_ids = set()
        
        for word in yes_variants:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            self.yes_token_ids.update(tokens)
        
        for word in no_variants:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            self.no_token_ids.update(tokens)
        
        # 合并所有相关 token
        self.yesno_token_ids = self.yes_token_ids.union(self.no_token_ids)
        
        if self.verbose:
            logging.info(f"Yes tokens: {len(self.yes_token_ids)}, No tokens: {len(self.no_token_ids)}")
    
    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """
        计算 Yes/No 输出的信息熵
        
        Args:
            logits: 模型输出的 logits (vocab_size,)
        
        Returns:
            entropy: 信息熵值
        """
        # 只关注 Yes/No 相关的 token
        if len(self.yesno_token_ids) == 0:
            # 如果没有识别到 Yes/No token，使用全词汇表
            probs = torch.softmax(logits, dim=-1)
        else:
            # 提取 Yes/No token 的 logits
            yesno_list = list(self.yesno_token_ids)
            yesno_logits = logits[yesno_list]
            probs = torch.softmax(yesno_logits, dim=-1)
        
        # 计算熵 H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        return entropy.item()
    
    def _build_probe_prompt(
        self,
        query_question: str,
        examples: List[Dict],
        k_round: int,
        system_prompt: str = None
    ) -> str:
        """
        构建探针 prompt
        
        Args:
            query_question: 查询问题
            examples: 当前轮的示例（前 k*n 个）
            k_round: 当前轮数 k
            system_prompt: 系统 prompt
        
        Returns:
            prompt_text: 完整的探针 prompt
        """
        if system_prompt is None:
            system_prompt = "You are a helpful math assistant."
        
        # 使用 paper 格式构建 prompt
        lines = []
        
        # 第一部分：只列题目（问题-only）
        num_examples = len(examples)
        problems_only_count = max(0, num_examples - self.paper_k_full)
        
        if problems_only_count > 0:
            lines.append("You will be provided Problems similar to the ones below:")
            for i in range(problems_only_count):
                q, _ = self.dataset_handler.format_example_cot(examples[i])
                lines.append(f"Problem: {q}")
            lines.append("—")
        
        # 第二部分：固定 K 条完整示例（问题-过程-答案）
        k_full = min(self.paper_k_full, num_examples)
        if k_full > 0:
            lines.append("Now, I am going to give you a series of demonstrations of Problems and Solutions to specify the output format.")
            lines.append("When you respond, think step by step, but your last line must be exactly of the form '#### <final_answer>'.")
            
            # 使用后面的 k_full 个示例作为完整示例
            start_idx = max(0, num_examples - k_full)
            for i in range(start_idx, num_examples):
                q, a_raw = self.dataset_handler.format_example_cot(examples[i])
                gold = self.dataset_handler.extract_gold_answer(examples[i])
                lines.append(f"Problem: {q}")
                lines.append("Solution:")
                lines.append(a_raw.rstrip())
                lines.append(f"Answer: {gold}.")
                lines.append(f"Final Answer: The final answer is {gold}.")
                lines.append("—")
        
        # 第三部分：目标问题
        lines.append(f"Problem: {query_question}")
        lines.append("—")
        
        # 第四部分：探针问题
        lines.append(self.probe_question)
        
        user_content = "\n".join(lines)
        
        # 构建 messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = f"{system_prompt}\n\n{user_content}\n"
        
        return prompt_text
    
    def _run_probe(
        self,
        query_question: str,
        examples: List[Dict],
        k_round: int,
        system_prompt: str = None
    ) -> Tuple[float, str, Dict]:
        """
        运行单次探针
        
        Args:
            query_question: 查询问题
            examples: 当前轮的示例
            k_round: 当前轮数
            system_prompt: 系统 prompt
        
        Returns:
            (entropy, response, details): 信息熵、模型响应、详细信息
        """
        # 构建 prompt
        prompt_text = self._build_probe_prompt(query_question, examples, k_round, system_prompt)
        
        # 编码
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt")
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        
        # 移动到正确的设备
        input_dev = self._pick_input_device()
        try:
            model_inputs = model_inputs.to(input_dev)
        except Exception:
            pass
        
        # 生成（只生成 1 个 token）
        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )
        
        # 获取第一个生成 token 的 logits
        first_token_logits = outputs.scores[0][0]  # (vocab_size,)
        
        # 计算熵
        entropy = self._compute_entropy(first_token_logits)
        
        # 解码响应
        full_seq = outputs.sequences[0]
        prompt_len = model_inputs["input_ids"].shape[1]
        gen_ids = full_seq[prompt_len:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # 详细信息
        details = {
            "round": k_round,
            "num_examples": len(examples),
            "entropy": entropy,
            "response": response,
            "threshold": self.entropy_threshold,
            "meets_threshold": entropy <= self.entropy_threshold
        }
        
        return entropy, response, details
    
    def _pick_input_device(self):
        """选择输入设备（处理分片情况）"""
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            device_map = self.model.hf_device_map
            first_layer = sorted(device_map.keys())[0]
            return device_map[first_layer]
        elif hasattr(self.model, 'device'):
            return self.model.device
        else:
            return torch.device('cpu')
    
    def determine_optimal_examples(
        self,
        query_question: str,
        sorted_examples: List[Dict],
        system_prompt: str = None
    ) -> Tuple[int, List[Dict]]:
        """
        通过迭代探针确定最优示例数量
        
        Args:
            query_question: 查询问题
            sorted_examples: BM25 排序后的示例列表
            system_prompt: 系统 prompt
        
        Returns:
            (optimal_k, probe_history): 最优示例数量和探针历史
        """
        probe_history = []
        
        # 从第一轮开始迭代
        for k in range(1, self.max_rounds + 1):
            # 当前轮激活的示例数量
            num_examples = k * self.window_size
            
            # 检查是否超出可用示例
            if num_examples > len(sorted_examples):
                if self.verbose:
                    logging.info(f"探针轮 {k}: 示例数 {num_examples} 超出可用范围 {len(sorted_examples)}，使用全部示例")
                num_examples = len(sorted_examples)
            
            # 获取当前示例
            current_examples = sorted_examples[:num_examples]
            
            # 运行探针
            entropy, response, details = self._run_probe(
                query_question, current_examples, k, system_prompt
            )
            
            probe_history.append(details)
            
            if self.verbose:
                logging.info(f"探针轮 {k}: 示例数={num_examples}, 熵={entropy:.4f}, "
                           f"阈值={self.entropy_threshold}, 响应='{response}'")
            
            # 检查是否满足阈值
            if entropy <= self.entropy_threshold:
                if self.verbose:
                    logging.info(f"✓ 探针轮 {k} 满足阈值，最优示例数: {num_examples}")
                return num_examples, probe_history
            
            # 检查是否已用尽所有示例
            if num_examples >= len(sorted_examples):
                if self.verbose:
                    logging.info(f"已用尽所有示例 ({len(sorted_examples)})，停止探针")
                return len(sorted_examples), probe_history
        
        # 达到最大轮数
        final_num = min(self.max_rounds * self.window_size, len(sorted_examples))
        if self.verbose:
            logging.info(f"达到最大轮数 {self.max_rounds}，使用示例数: {final_num}")
        
        return final_num, probe_history


def test_probe_mechanism():
    """测试探针机制"""
    import sys
    sys.path.insert(0, '..')
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from util.dataset_handlers import get_dataset_handler
    
    logging.basicConfig(level=logging.INFO)
    
    # 加载模型
    model_path = "/home/models/models/Qwen/Qwen2.5-14B-Instruct"
    logging.info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    )
    model.eval()
    
    # 获取数据集处理器
    dataset_handler = get_dataset_handler("openai/gsm8k", "main")
    train, _ = dataset_handler.load_and_split(test_size=100, seed=42)
    
    # 创建探针机制
    probe = ProbeMechanism(
        model=model,
        tokenizer=tokenizer,
        probe_question="Based on the above examples, are you confident to answer this question? Please answer Yes or No.",
        entropy_threshold=0.5,
        window_size=2,
        max_rounds=5,
        dataset_handler=dataset_handler
    )
    
    # 测试
    test_examples = train.select(range(10))
    test_question = "John has 10 apples. He gives 3 to Mary. How many apples does John have now?"
    
    optimal_k, history = probe.determine_optimal_examples(test_question, list(test_examples))
    
    print(f"\n最优示例数: {optimal_k}")
    print(f"探针历史 ({len(history)} 轮):")
    for h in history:
        print(f"  轮 {h['round']}: {h['num_examples']} 示例, 熵={h['entropy']:.4f}, 响应='{h['response']}'")


if __name__ == "__main__":
    test_probe_mechanism()
