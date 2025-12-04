"""
探针选择器
按窗口大小 n 逐轮扩展 shots，通过熵判断置信度

流程:
1. 按 ranked_shots 顺序，每轮引入 n 个新 shot
2. 使用 KV cache 构造 probe 输入: concat(K_shots, K_prompt_query)
3. 生成 1 个 token，计算输出分布的熵
4. 如果熵 < 阈值，认为当前 shots 足够，停止扩展
5. 否则继续下一轮
"""
import sys
import os
import torch
import logging
import numpy as np
from typing import List, Tuple, Dict
from tqdm.auto import tqdm

# 添加 util 路径（复制自 baseline 的通用模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'util'))

from kv_pool_manager import KVPoolManager


class ProbeSelector:
    """探针选择器"""
    
    # 🔥 探针问题：让模型回答 Yes/No，限制输出在单 token
    PROBE_QUESTION = "Based on the above examples, are you confident to answer this question? Please answer Yes or No."
    
    def __init__(
        self,
        model,
        tokenizer,
        kv_pool_manager: KVPoolManager,
        window_size: int = 4,
        entropy_threshold: float = 0.5,
        max_rounds: int = 256,
        device: str = "npu",
        verbose: bool = True,
        mode: str = "cot",  # 添加模式参数
        paper_num_questions: int = 4,  # Paper 模式的 question-only shots 数量
        output_dir: str = None  # 输出目录
    ):
        """
        初始化探针选择器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            kv_pool_manager: KV 池管理器
            window_size: 每轮引入的 shot 数量 (n)
            entropy_threshold: 熵阈值 (τ)
            max_rounds: 最大探针轮数
            device: 计算设备
            verbose: 是否输出详细日志
            mode: 模式 ('cot', 'io', 'paper')
            paper_num_questions: Paper 模式的 question-only shots 数量
            output_dir: 输出目录（用于保存探针 prompt 示例）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kv_pool = kv_pool_manager
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.max_rounds = max_rounds
        self.device = device
        self.verbose = verbose
        self.mode = mode
        self.paper_num_questions = paper_num_questions
        self.output_dir = output_dir
        
        # 🔥 标志位：只保存一次探针 prompt 示例
        self._probe_example_saved = False
        
        # 模型配置
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_key_value_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        # 🔥 预编译 Yes/No token IDs
        self._prepare_yesno_tokens()
        
        logging.info(f"初始化 ProbeSelector: window_size={window_size}, "
                    f"entropy_threshold={entropy_threshold}, max_rounds={max_rounds}, mode={mode}")
    
    def _prepare_yesno_tokens(self):
        """预编译 Yes/No 的 token IDs"""
        yes_variants = ["Yes", "yes", "YES", " Yes", " yes"]
        no_variants = ["No", "no", "NO", " No", " no"]
        
        self.yes_token_ids = set()
        self.no_token_ids = set()
        
        for word in yes_variants:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            self.yes_token_ids.update(tokens)
        
        for word in no_variants:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            self.no_token_ids.update(tokens)
        
        self.yesno_token_ids = list(self.yes_token_ids.union(self.no_token_ids))
        
        if self.verbose:
            logging.info(f"Yes tokens: {self.yes_token_ids}, No tokens: {self.no_token_ids}")
    
    def _get_prompt_query_kv(self, prompt_tokens: torch.Tensor, query_tokens: torch.Tensor):
        """获取 prompt+query 的 KV"""
        input_ids = torch.cat([prompt_tokens, query_tokens], dim=-1)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True
            )
            K_layers, V_layers = self.kv_pool._extract_kv_from_model_output(outputs)
        
        return K_layers, V_layers
    
    def _reshape_kv_to_past_format(self, K_layers, V_layers):
        """
        将简化的 KV 转换为 past_key_values 格式
        
        Args:
            K_layers: List[K_layer], 每个 K_layer: [T, d_k]
            V_layers: List[V_layer], 每个 V_layer: [T, d_v]
        
        Returns:
            past_key_values: tuple of (key, value) pairs
        """
        past_key_values = []
        
        for layer_idx in range(len(K_layers)):
            K = K_layers[layer_idx]  # [T, d_k]
            V = V_layers[layer_idx]  # [T, d_v]
            
            seq_len = K.shape[0]
            assert K.shape[1] == self.num_key_value_heads * self.head_dim, (
                f"K dim mismatch at layer {layer_idx}: {K.shape[1]} vs {self.num_key_value_heads * self.head_dim}")
            assert V.shape[1] == self.num_key_value_heads * self.head_dim, (
                f"V dim mismatch at layer {layer_idx}: {V.shape[1]} vs {self.num_key_value_heads * self.head_dim}")
            assert K.shape[0] == V.shape[0], (
                f"Seq len mismatch at layer {layer_idx}: K={K.shape[0]} V={V.shape[0]}")
            K = K.contiguous()
            V = V.contiguous()
            
            # Reshape: [T, num_key_value_heads * head_dim] -> [1, num_key_value_heads, T, head_dim]
            K_reshaped = K.view(seq_len, self.num_key_value_heads, self.head_dim).unsqueeze(0)
            K_reshaped = K_reshaped.permute(0, 2, 1, 3)  # [1, num_key_value_heads, T, head_dim]
            
            V_reshaped = V.view(seq_len, self.num_key_value_heads, self.head_dim).unsqueeze(0)
            V_reshaped = V_reshaped.permute(0, 2, 1, 3)
            
            past_key_values.append((K_reshaped, V_reshaped))
        
        return tuple(past_key_values)
    
    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """
        计算 Yes/No 输出分布的熵
        
        Args:
            logits: 输出 logits [vocab_size]
        
        Returns:
            entropy: 熵值
        """
        # 检查 logits 是否有效
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.warning("⚠️ logits 包含 NaN 或 Inf，返回默认高熵值")
            return 10.0
        
        # 🔥 只关注 Yes/No 相关的 token
        if len(self.yesno_token_ids) > 0:
            yesno_logits = logits[self.yesno_token_ids]
            probs = torch.softmax(yesno_logits.float(), dim=-1)
        else:
            # fallback: 使用全词汇表
            probs = torch.softmax(logits.float(), dim=-1)
        
        # 计算熵 H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        
        # 最后检查一次
        if not torch.isfinite(torch.tensor(entropy)):
            logging.warning(f"⚠️ 熵值计算结果异常: {entropy}，返回默认高熵值")
            return 10.0
        
        return entropy
    
    def _get_kv_excluding_last_token(self, text: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        为文本计算 KV cache，但排除最后一个 token
        返回 KV cache 和最后一个 token 的 ID
        
        Args:
            text: 输入文本
        
        Returns:
            (K_layers, V_layers, last_token_id): 不含最后一个 token 的 KV 和最后一个 token ID
        """
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')[0]
        
        if len(tokens) <= 1:
            # 如果只有一个 token，返回空 KV 和该 token
            last_token_id = tokens[-1].item() if len(tokens) > 0 else self.tokenizer.eos_token_id
            empty_K = [torch.zeros((0, self.kv_pool.d_k), device=self.device) for _ in range(self.num_layers)]
            empty_V = [torch.zeros((0, self.kv_pool.d_v), device=self.device) for _ in range(self.num_layers)]
            return empty_K, empty_V, last_token_id
        
        # 保存最后一个 token ID
        last_token_id = tokens[-1].item()
        
        # 为不含最后一个 token 的序列计算 KV
        tokens_without_last = tokens[:-1]
        
        with torch.inference_mode():
            input_ids = tokens_without_last.unsqueeze(0).to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True
            )
            K_layers, V_layers = self.kv_pool._extract_kv_from_model_output(outputs)
        
        return K_layers, V_layers, last_token_id
    
    def _run_probe(
        self,
        prompt_tokens: torch.Tensor,
        candidate_shot_ids: List[int],
        query_tokens: torch.Tensor,
        query_text: str = "",  # 添加 query 文本参数用于构建 prompt
        query_kv_cache: Tuple = None,  # 预计算的 query KV cache（不含最后一个 token）
        start_token_id: int = None  # 预计算的起始 token ID
    ) -> Tuple[float, str]:
        """
        运行一次探针（使用 KV cache，支持不同模式）
        
        Args:
            prompt_tokens: Prompt token IDs
            candidate_shot_ids: 候选 shot ID 列表
            query_tokens: Query token IDs
            query_text: Query 文本（用于构建格式化 prompt）
            query_kv_cache: 预计算的 query KV cache（不含最后一个 token）
            start_token_id: 起始 token ID（query 的最后一个 token）
        
        Returns:
            (entropy, response): 熵值和生成的 token 文本
        """
        # ✅ 步骤 0: 获取固定部分的 KV（System Prompt 或 Paper 固定部分）
        if self.mode == "paper":
            # Paper 模式：使用预缓存的固定部分（引导语 + fullshots）
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('paper_fixed')
        else:
            # CoT/IO 模式：使用 System Prompt
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('system_prompt')
        
        # ✅ 步骤 1: 根据模式获取 shots 的 KV
        if self.mode == "paper":
            # Paper 模式：只需要 question-only shots
            # 注意：paper 的 fullshots 已经在固定部分了
            K_shots_layers, V_shots_layers = self._get_paper_question_only_kv(candidate_shot_ids)
        else:
            # CoT/IO 模式：直接使用预计算的 KV
            K_shots_layers, V_shots_layers = self.kv_pool.get_kv_for_shots(candidate_shot_ids)
        
        # 2. 获取 query + 探针问题 的 KV（使用预计算的缓存或现场计算）
        # 🔥 关键修复：query KV 不包含最后一个 token，最后一个 token 作为 start_token 输入
        if query_kv_cache is not None:
            # 使用预计算的 query KV cache（已排除最后一个 token）
            K_query_layers, V_query_layers = query_kv_cache
        elif query_text:
            # 🔥 Fallback: 使用探针问题格式
            formatted_query = f"Problem: {query_text}\n{self.PROBE_QUESTION}"
            K_query_layers, V_query_layers, start_token_id = self._get_kv_excluding_last_token(formatted_query)
        else:
            # 使用原始 tokens
            K_query_layers, V_query_layers = self._get_prompt_query_kv(prompt_tokens, query_tokens)
        
        # ✅ 步骤 3: 拼接所有部分：固定部分 + shots + query
        K_all_layers = []
        V_all_layers = []
        for layer_idx in range(self.num_layers):
            parts_K = []
            parts_V = []
            
            # 1. 固定部分
            if K_fixed_layers[layer_idx].shape[0] > 0:
                parts_K.append(K_fixed_layers[layer_idx])
                parts_V.append(V_fixed_layers[layer_idx])
            
            # 2. Shots
            if K_shots_layers[layer_idx].shape[0] > 0:
                parts_K.append(K_shots_layers[layer_idx])
                parts_V.append(V_shots_layers[layer_idx])
            
            # 3. Query
            if K_query_layers[layer_idx].shape[0] > 0:
                parts_K.append(K_query_layers[layer_idx])
                parts_V.append(V_query_layers[layer_idx])
            
            # 拼接
            K_all = torch.cat(parts_K, dim=0) if parts_K else torch.zeros((0, self.kv_pool.d_k), device=self.device)
            V_all = torch.cat(parts_V, dim=0) if parts_V else torch.zeros((0, self.kv_pool.d_v), device=self.device)
            
            K_all_layers.append(K_all)
            V_all_layers.append(V_all)
        
        # 4. 转换为 past_key_values 格式
        print(f"[DEBUG-PROBE] 开始转换 past_key_values, 层数={len(K_all_layers)}, seq_len={K_all_layers[0].shape[0] if K_all_layers else 0}")
        past_key_values = self._reshape_kv_to_past_format(K_all_layers, V_all_layers)
        print(f"[DEBUG-PROBE] past_key_values 转换完成, shape={past_key_values[0][0].shape}")
        
        # 5. 生成 1 个token - 使用手动 forward 而不是 generate
        # 🔥 使用预计算的 start_token_id，避免重复
        if start_token_id is not None:
            start_token = torch.tensor([[start_token_id]], dtype=torch.long, device=self.device)
        else:
            # fallback: 使用 query 的最后一个 token
            start_token = query_tokens[-1:].long().unsqueeze(0).to(self.device)
        
        # 计算 past_key_values 的长度（已经缓存的 tokens 数量）
        past_length = K_all_layers[0].shape[0] if K_all_layers else 0
        
        # 创建 attention_mask: 所有 past tokens + 当前 input token 都是有效的
        attention_mask = torch.ones((1, past_length + 1), dtype=torch.long, device=self.device)
        
        with torch.inference_mode():
            # 手动 forward 获取 logits
            print(f"[DEBUG-PROBE] 准备 forward, start_token shape={start_token.shape}, attention_mask shape={attention_mask.shape}")
            outputs = self.model(
                input_ids=start_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            print(f"[DEBUG-PROBE] forward 完成")
            
            # 获取最后一个位置的 logits
            first_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
            
            # 计算熵
            entropy = self._compute_entropy(first_token_logits)
            
            # 🔥 在 Yes/No tokens 中选择最可能的 token（而不是全词汇表）
            if len(self.yesno_token_ids) > 0:
                yesno_logits = first_token_logits[self.yesno_token_ids]
                best_idx = torch.argmax(yesno_logits).item()
                next_token_id = self.yesno_token_ids[best_idx]
            else:
                next_token_id = torch.argmax(first_token_logits).item()
            
            response = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
            
            # 输出调试信息
            print(f"[DEBUG-PROBE] next_token_id={next_token_id}, response='{response}'")
        
        return entropy, response
    
    def _get_paper_question_only_kv(self, shot_ids: List[int]):
        """
        为 Paper 模式获取 question-only shots 的 KV
        
        Args:
            shot_ids: Shot ID 列表
        
        Returns:
            (K_layers, V_layers): 拼装好的 KV
        """
        # 构建 question-only部分
        parts = []
        if shot_ids:
            parts.append("You will be provided Problems similar to the ones below:")
            for sid in shot_ids:
                if sid in self.kv_pool.kv_cache_pool:
                    example = self.kv_pool.kv_cache_pool[sid]['example']
                    q, _ = self.kv_pool.dataset_handler.format_example_cot(example)
                    parts.append(f"Problem: {q}")
            parts.append("—")  # 分隔符
        
        if parts:
            combined_text = "\n".join(parts)
            K_layers, V_layers = self.kv_pool.get_kv_for_text(combined_text)
        else:
            # 返回空 KV
            K_layers = [torch.zeros((0, self.kv_pool.d_k), device=self.device) for _ in range(self.num_layers)]
            V_layers = [torch.zeros((0, self.kv_pool.d_v), device=self.device) for _ in range(self.num_layers)]
        
        return K_layers, V_layers
    
    def _save_probe_prompt_example(self, query_text: str, shot_ids: List[int]):
        """
        保存完整的探针 prompt 示例文件供审查（只保存一次）
        
        Args:
            query_text: 查询文本
            shot_ids: Shot ID 列表
        """
        # 🔥 只保存一次
        if self._probe_example_saved:
            return
        
        try:
            import os
            from datetime import datetime
            
            # 构建完整的 prompt
            parts = []
            
            # 1. System Prompt
            system_prompt = self.kv_pool.get_fixed_text('system_prompt')
            if system_prompt:
                parts.append("=" * 60)
                parts.append("[SYSTEM PROMPT]")
                parts.append("=" * 60)
                parts.append(system_prompt)
            
            # 2. Shots
            parts.append("\n" + "=" * 60)
            parts.append(f"[SHOTS ({len(shot_ids)} 个)]")
            parts.append("=" * 60)
            for i, sid in enumerate(shot_ids):
                if sid in self.kv_pool.kv_cache_pool:
                    example = self.kv_pool.kv_cache_pool[sid]['example']
                    q, a = self.kv_pool.dataset_handler.format_example_cot(example)
                    parts.append(f"\n--- Shot {i+1} (ID={sid}) ---")
                    parts.append(f"Problem: {q}")
                    parts.append(f"Solution: {a}")
            
            # 3. Query + 探针问题
            parts.append("\n" + "=" * 60)
            parts.append("[QUERY + 探针问题]")
            parts.append("=" * 60)
            parts.append(f"Problem: {query_text}")
            parts.append(f"\n{self.PROBE_QUESTION}")
            
            # 🔥 使用配置的输出目录，如果没有则使用脚本目录
            if self.output_dir:
                save_dir = self.output_dir
            else:
                save_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(save_dir, f"probe_prompt_example_{timestamp}.txt")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(parts))
            
            # 🔥 标记已保存
            self._probe_example_saved = True
            logging.info(f"📄 完整探针 prompt 已保存到: {filepath}")
        except Exception as e:
            logging.warning(f"⚠️ 保存探针 prompt 失败: {e}")
    
    def select_shots_with_probe(
        self,
        ranked_shots: List[int],
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        query_text: str = ""  # 添加 query 文本参数
    ) -> Tuple[List[int], List[Dict]]:
        """
        使用探针机制选择 shots
        
        Args:
            ranked_shots: 排序后的 shot ID 列表
            prompt_tokens: Prompt token IDs
            query_tokens: Query token IDs
            query_text: Query 文本（用于格式化）
        
        Returns:
            (selected_shots, probe_history): 选中的 shot 列表和探针历史
        """
        selected_shots = []
        probe_history = []
        
        idx = 0
        round_num = 0
        
        # 🔥 优化:预计算 query + 探针问题 的 KV(不含最后一个 token),避免每轮重复计算
        query_kv_cache = None
        start_token_id = None  # 保存最后一个 token ID
        if query_text and self.mode in ["cot", "io", "paper"]:
            if self.verbose:
                logging.info(f"📌 预计算 query + 探针问题 KV cache...")
            try:
                # 🔥 构建包含探针问题的完整 query
                # 模型看到的内容：[System Prompt] + [Shots] + [Query + 探针问题]
                formatted_query = f"Problem: {query_text}\n{self.PROBE_QUESTION}"
                
                # 获取不含最后一个 token 的 KV
                K_layers, V_layers, start_token_id = self._get_kv_excluding_last_token(formatted_query)
                query_kv_cache = (K_layers, V_layers)
                
                # 重新 tokenize 以获取正确的 query_tokens
                query_tokens = self.tokenizer.encode(formatted_query, add_special_tokens=False, return_tensors='pt')[0]
                if self.verbose:
                    logging.info(f"Query+探针问题 tokens 长度: {len(query_tokens)}, KV 长度: {K_layers[0].shape[0]}")
                    logging.info(f"✓ Query KV cache 预计算完成, start_token_id={start_token_id} ('{self.tokenizer.decode([start_token_id])}')") 
            except Exception as e:
                logging.warning(f"⚠️ Query KV 预计算失败,将在每轮重新计算: {e}")
                import traceback
                traceback.print_exc()
                query_kv_cache = None
                start_token_id = None
        
        if self.verbose:
            logging.info(f"开始探针选择（{self.mode} 模式），总共 {len(ranked_shots)} 个候选 shot...")
        
        # 注意：探针 prompt 示例在探针完成后保存，以展示最终选定的 shots
        
        while idx < len(ranked_shots) and round_num < self.max_rounds:
            round_num += 1
            
            # 本轮新引入的 shots
            new_shots = ranked_shots[idx : idx + self.window_size]
            candidate_shots = selected_shots + new_shots
            
            # 运行探针，传入预计算的 query KV 和 start_token_id
            print(f"[DEBUG-PROBE] 轮 {round_num}: 准备运行探针, candidate_shots={len(candidate_shots)}")
            entropy, response = self._run_probe(
                prompt_tokens, candidate_shots, query_tokens, query_text, query_kv_cache, start_token_id
            )
            print(f"[DEBUG-PROBE] 轮 {round_num}: 探针完成, entropy={entropy:.4f}")
            
            # 记录历史
            probe_record = {
                'round': round_num,
                'num_shots': len(candidate_shots),
                'new_shots': new_shots,
                'entropy': entropy,
                'response': response,
                'threshold': self.entropy_threshold,
                'meets_threshold': entropy < self.entropy_threshold,
                'mode': self.mode
            }
            probe_history.append(probe_record)
            
            if self.verbose:
                logging.info(f"探针轮 {round_num}: shots={len(candidate_shots)}, "
                           f"entropy={entropy:.4f}, threshold={self.entropy_threshold}, "
                           f"response='{response[:20]}'")
            
            # 判断是否满足阈值
            if entropy < self.entropy_threshold:
                selected_shots = candidate_shots
                if self.verbose:
                    logging.info(f"✓ 探针轮 {round_num} 满足阈值，最终选中 {len(selected_shots)} 个 shot")
                break
            else:
                # 不够好，但仍然加入候选
                selected_shots = candidate_shots
                idx += self.window_size
        
        # 如果跑完所有轮还没达到阈值
        if round_num >= self.max_rounds or idx >= len(ranked_shots):
            if self.verbose:
                logging.info(f"达到最大轮数或用尽所有 shot，最终选中 {len(selected_shots)} 个 shot")
        
        # 🔥 探针完成后，保存最终选定的 shots 对应的 prompt 示例
        self._save_probe_prompt_example(query_text, selected_shots)
        
        return selected_shots, probe_history
