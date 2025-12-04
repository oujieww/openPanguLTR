"""KV 拼装器
负责将选中 shots 的 KV cache 与 prompt+query 的 KV 拼装成最终序列

逻辑序列:
[prompt_tokens, shot_a, shot_b, ..., shot_k, prompt_tokens, query_tokens]

KV 拼装:
K_final_layer = concat(K_shots_layer, K_prompt_query_layer, dim=0) 为每一层
V_final_layer = concat(V_shots_layer, V_prompt_query_layer, dim=0)

支持三种模式: CoT, IO, Paper

支持两种 KV 复用策略:
1. 文本拼接模式 (use_text_forward=True): 安全但慢
2. KV 复用模式 (use_kv_reuse=True): 快速，使用 RoPE 位置校正
"""
import sys
import os
import torch
import logging
import copy
from typing import List, Tuple, Optional, Dict

# 添加根路径以定位 util 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kv_pool_manager import KVPoolManager
from rope_utils import RoPECorrector  # 🔥 新增: RoPE 位置校正器


class KVAssembler:
    """KV 拼装器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        kv_pool_manager: KVPoolManager,
        device: str = "npu",
        mode: str = "cot",
        paper_num_questions: int = 4,
        use_text_forward: bool = False,  # 文本拼接模式（安全但慢）
        use_kv_reuse: bool = True  # 🔥 KV 复用模式（快速，使用 RoPE 校正）
    ):
        """
        初始化 KV 拼装器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            kv_pool_manager: KV 池管理器
            device: 计算设备
            mode: 模式 ('cot', 'io', 'paper')
            paper_num_questions: Paper 模式的 question-only shots 数量
            use_text_forward: 文本拼接模式（每次重新计算所有 KV，安全但慢）
            use_kv_reuse: KV 复用模式（复用预计算的 KV + RoPE 位置校正，快速）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kv_pool = kv_pool_manager
        self.device = device
        self.mode = mode
        self.paper_num_questions = paper_num_questions
        self.use_text_forward = use_text_forward
        self.use_kv_reuse = use_kv_reuse  # 🔥 KV 复用模式
        self.use_hybrid_mode = False  # 禁用旧模式
        self._last_part_sizes = {"fixed": 0, "shots": 0, "query": 0}
        
        # 模型配置
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_key_value_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        # 🔥 初始化 RoPE 校正器
        if use_kv_reuse:
            self.rope_corrector = RoPECorrector(model, device)
            logging.info("✅ RoPE 校正器已初始化，启用 KV 复用模式")
        else:
            self.rope_corrector = None
        
        logging.info(f"初始化 KVAssembler: num_layers={self.num_layers}, "
                    f"num_heads={self.num_heads}, head_dim={self.head_dim}, mode={mode}")
    
    def _get_layer_device(self, layer_idx: int):
        """
        获取指定层的设备
        支持模型分片（device_map='auto'）
        
        Args:
            layer_idx: 层索引
        
        Returns:
            device: 该层所在的设备
        """
        # 获取该层的设备
        layer_module = self.model.model.layers[layer_idx]
        # 从层的第一个参数获取设备
        return next(layer_module.parameters()).device
    
    def _get_prompt_query_kv(
        self,
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        获取 prompt + query 的 KV
        
        Args:
            prompt_tokens: Prompt token IDs
            query_tokens: Query token IDs
        
        Returns:
            (K_layers, V_layers): 所有层的 KV
        """
        # 拼接 prompt 和 query
        input_ids = torch.cat([prompt_tokens, query_tokens], dim=-1).contiguous().clone()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        # 确保是整数类型（NPU要求）
        input_ids = input_ids.long().to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True
            )
            
            # 提取 KV (所有层)
            K_layers, V_layers = self.kv_pool._extract_kv_from_model_output(outputs)
        
        return K_layers, V_layers
    
    def _reshape_kv_to_past_format(
        self,
        K_layers: List[torch.Tensor],
        V_layers: List[torch.Tensor]
    ) -> Tuple:
        """
        将简化的 KV 转换为 past_key_values 格式
        
        past_key_values 格式: tuple of num_layers
        每层: (key, value)
        key/value shape: [batch, num_heads, seq_len, head_dim]
        
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
            V_reshaped = V_reshaped.permute(0, 2, 1, 3)  # [1, num_key_value_heads, T, head_dim]
            
            past_key_values.append((K_reshaped, V_reshaped))
        
        return tuple(past_key_values)
    
    def assemble_kv_for_generation(
        self,
        selected_shot_ids: List[int],
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        query_text: str = ""  # 添加 query 文本参数
    ) -> Tuple:
        """
        拼装最终的 KV cache 用于生成（支持不同模式）
        
        Args:
            selected_shot_ids: 选中的 shot ID 列表
            prompt_tokens: Prompt token IDs
            query_tokens: Query token IDs
            query_text: Query 文本（用于格式化）
        
        Returns:
            past_key_values: 拼装好的 KV cache
        """
        logging.info(f"开始拼装 KV cache（{self.mode} 模式），共 {len(selected_shot_ids)} 个 shot...")
        print(f"[DEBUG-KV] 开始拼装, mode={self.mode}, shots={len(selected_shot_ids)}")
        
        # ✅ 步骤 0: 获取固定部分的 KV（System Prompt 或 Paper 固定部分）
        if self.mode == "paper":
            # Paper 模式：使用预缓存的固定部分（引导语 + fullshots）
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('paper_fixed')
            logging.info(f"✅ 使用 Paper 固定部分 KV，tokens: {K_fixed_layers[0].shape[0] if K_fixed_layers[0].shape[0] > 0 else 0}")
        else:
            # CoT/IO 模式：使用 System Prompt
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('system_prompt')
            logging.info(f"✅ 使用 System Prompt KV，tokens: {K_fixed_layers[0].shape[0] if K_fixed_layers[0].shape[0] > 0 else 0}")
        
        # ✅ 步骤 1: 获取动态选中的 shots KV
        if self.mode == "paper":
            # Paper 模式：只需要 question-only shots
            # 注意：paper 的 fullshots 已经在固定部分了，这里只需要额外的 question-only
            K_shots_layers, V_shots_layers = self._get_paper_question_only_kv(selected_shot_ids)
        else:
            # CoT/IO 模式：直接使用预计算的 KV
            K_shots_layers, V_shots_layers = self.kv_pool.get_kv_for_shots(selected_shot_ids)
            logging.info(f"✅ 直接使用预计算的 KV cache，无需重新计算")
        
        logging.info(f"Shots KV: {len(K_shots_layers)} 层, 每层 K shape={K_shots_layers[0].shape}")
        
        # ✅ 步骤 2: 获取 query 的 KV（根据模式格式化）
        if query_text and self.mode in ["cot", "io"]:
            header = "When you respond, your last line must be exactly of the form '#### <final_answer>'."
            if self.mode == "cot":
                formatted_query = f"{header}\nProblem: {query_text}\nSolution:"
            else:
                formatted_query = f"{header}\nProblem: {query_text}\nAnswer:"
            K_query_layers, V_query_layers = self.kv_pool.get_kv_for_text(formatted_query)
        elif self.mode == "paper" and query_text:
            header = "When you respond, your last line must be exactly of the form '#### <final_answer>'."
            formatted_query = f"{header}\nProblem: {query_text}\nSolution:"
            K_query_layers, V_query_layers = self.kv_pool.get_kv_for_text(formatted_query)
        else:
            # 使用原始 tokens
            K_query_layers, V_query_layers = self._get_prompt_query_kv(prompt_tokens, query_tokens)
        
        logging.info(f"Query KV: {len(K_query_layers)} 层, 每层 K shape={K_query_layers[0].shape}")

        # 记录各部分 token 数（以第 0 层为准）
        try:
            self._last_part_sizes = {
                "fixed": int(K_fixed_layers[0].shape[0]),
                "shots": int(K_shots_layers[0].shape[0]),
                "query": int(K_query_layers[0].shape[0])
            }
        except Exception:
            self._last_part_sizes = {"fixed": 0, "shots": 0, "query": 0}
        
        # ✅ 步骤 3: 拼接所有部分：固定部分 + shots + query
        # 🔥 NPU 优化：先移至 CPU 拼接，再传回对应层的 NPU，避免 ConcatD 内存错误
        K_all_layers = []
        V_all_layers = []
        for layer_idx in range(self.num_layers):
            parts_K = []
            parts_V = []
            
            # 1. 固定部分 (System Prompt 或 Paper 固定部分)
            if K_fixed_layers[layer_idx].shape[0] > 0:
                parts_K.append(K_fixed_layers[layer_idx].cpu())  # 移至 CPU
                parts_V.append(V_fixed_layers[layer_idx].cpu())
            
            # 2. Shots
            if K_shots_layers[layer_idx].shape[0] > 0:
                parts_K.append(K_shots_layers[layer_idx].cpu())  # 移至 CPU
                parts_V.append(V_shots_layers[layer_idx].cpu())
            
            # 3. Query
            if K_query_layers[layer_idx].shape[0] > 0:
                parts_K.append(K_query_layers[layer_idx].cpu())  # 移至 CPU
                parts_V.append(V_query_layers[layer_idx].cpu())
            
            # 获取该层的实际设备（支持分片模型）
            layer_device = self._get_layer_device(layer_idx)
            
            # 在 CPU 上拼接（避免 NPU ConcatD 错误）
            K_final = torch.cat(parts_K, dim=0).contiguous().clone()  # CPU 上拼接
            K_final = K_final.to(layer_device)  # 传回该层的 NPU
            
            V_final = torch.cat(parts_V, dim=0).contiguous().clone()  # CPU 上拼接
            V_final = V_final.to(layer_device)  # 传回该层的 NPU
            
            K_all_layers.append(K_final)
            V_all_layers.append(V_final)
        
        logging.info(f"最终 KV: {len(K_all_layers)} 层, 每层 K shape={K_all_layers[0].shape}")
        
        # 4. 转换为 past_key_values 格式
        past_key_values = self._reshape_kv_to_past_format(K_all_layers, V_all_layers)
        print("[KV] KV cache 拼接完成:", past_key_values[0][0].shape)
        return past_key_values
    
    def generate_with_kv_cache(
        self,
        selected_shot_ids: List[int],
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        max_new_tokens: int = 512,
        query_text: str = "",
        **gen_kwargs
    ) -> Tuple[str, Dict]:
        """
        使用拼装的 KV cache 生成答案
        
        Args:
            selected_shot_ids: 选中的 shot ID 列表
            prompt_tokens: Prompt token IDs
            query_tokens: Query token IDs
            max_new_tokens: 最大生成 token 数
            query_text: Query 文本（用于格式化）
            **gen_kwargs: 其他生成参数
        
        Returns:
            (response, gen_info): 生成的文本和生成信息
        """
        # 🔥 KV 复用模式（复用预计算的 KV + RoPE 位置校正）
        if self.use_kv_reuse and self.rope_corrector is not None:
            return self._generate_with_kv_reuse(
                selected_shot_ids, prompt_tokens, query_tokens,
                max_new_tokens, query_text, **gen_kwargs
            )
        
        # 文本拼接模式（安全但慢）
        if self.use_text_forward:
            return self._generate_with_text_forward(
                selected_shot_ids, prompt_tokens, query_tokens,
                max_new_tokens, query_text, **gen_kwargs
            )
        
        # 旧的混合模式（已禁用）
        if self.use_hybrid_mode:
            return self._generate_with_hybrid_mode(
                selected_shot_ids, prompt_tokens, query_tokens,
                max_new_tokens, query_text, **gen_kwargs
            )
        
        # 默认：文本拼接模式
        return self._generate_with_text_forward(
            selected_shot_ids, prompt_tokens, query_tokens,
            max_new_tokens, query_text, **gen_kwargs
        )
    
    def _generate_with_text_forward(
        self,
        selected_shot_ids: List[int],
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        max_new_tokens: int = 512,
        query_text: str = "",
        **gen_kwargs
    ) -> Tuple[str, Dict]:
        """
        🔥 使用文本拼接模式生成（确保位置编码正确）
        
        将所有内容拼接成完整文本后一次性 forward，避免 KV 拼接导致的位置编码错误。
        """
        # 1. 构建完整的 prompt 文本
        full_prompt_parts = []
        
        # 1.1 固定部分（System Prompt 或 Paper 固定部分）
        if self.mode == "paper":
            fixed_text = self.kv_pool.get_fixed_text('paper_fixed')
        else:
            fixed_text = self.kv_pool.get_fixed_text('system_prompt')
        if fixed_text:
            full_prompt_parts.append(fixed_text)
        
        # 1.2 Shots 部分
        if selected_shot_ids:
            if self.mode == "paper":
                # Paper 模式：question-only
                shot_parts = ["You will be provided Problems similar to the ones below:"]
                for sid in selected_shot_ids:
                    if sid in self.kv_pool.kv_cache_pool:
                        example = self.kv_pool.kv_cache_pool[sid]['example']
                        q, _ = self.kv_pool.dataset_handler.format_example_cot(example)
                        shot_parts.append(f"Problem: {q}")
                shot_parts.append("—")
                full_prompt_parts.append("\n".join(shot_parts))
            else:
                # CoT/IO 模式：完整的 shot
                fmt = "cot" if self.mode == "cot" else "io"
                shots_text = "\n".join([
                    self.kv_pool.format_shot_text(sid, fmt) 
                    for sid in selected_shot_ids 
                    if sid in self.kv_pool.kv_cache_pool
                ])
                full_prompt_parts.append(shots_text)
        
        # 1.3 Query 部分
        header = "When you respond, your last line must be exactly of the form '#### <final_answer>'."
        if self.mode == "cot" or self.mode == "paper":
            formatted_query = f"{header}\nProblem: {query_text}\nSolution:"
        else:
            formatted_query = f"{header}\nProblem: {query_text}\nAnswer:"
        full_prompt_parts.append(formatted_query)
        
        # 2. 拼接完整 prompt
        full_prompt = "\n".join([p for p in full_prompt_parts if p]).strip()
        
        # 3. Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.long() if k == 'input_ids' else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        prompt_len = inputs["input_ids"].shape[1]
        print(f"[GEN-TEXT] 完整 prompt 长度: {prompt_len} tokens")
        
        # 4. 生成
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                use_cache=True
            )
        
        # 5. 解码生成的部分
        generated_ids = outputs.sequences[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"[GEN-TEXT] 生成完成, 输出 {len(generated_ids)} tokens")
        
        # 6. 生成信息
        gen_info = {
            'num_shots_used': len(selected_shot_ids),
            'total_kv_layers': self.num_layers,
            'total_kv_tokens': prompt_len,
            'output_tokens': len(generated_ids),
            'mode': f"{self.mode}_text_forward",
            'kv_tokens_per_part': {"fixed": 0, "shots": 0, "query": prompt_len}
        }
        
        return response, gen_info
    
    def _generate_with_kv_reuse(
        self,
        selected_shot_ids: List[int],
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        max_new_tokens: int = 512,
        query_text: str = "",
        **gen_kwargs
    ) -> Tuple[str, Dict]:
        """
        🔥 KV 复用模式：复用预计算的所有 KV cache + RoPE 位置校正
        
        这是最高效的模式：
        1. System Prompt KV: 复用（位置 0 开始，无需校正）
        2. Shots KV: 复用 + RoPE 位置校正（调整到正确的绝对位置）
        3. Query: 只计算 Query 部分的 KV
        """
        # ============ 步骤 1: 获取固定部分的 KV cache （已预计算，位置从 0 开始） ============
        if self.mode == "paper":
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('paper_fixed')
        else:
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('system_prompt')
        
        fixed_length = K_fixed_layers[0].shape[0] if K_fixed_layers[0].shape[0] > 0 else 0
        current_offset = fixed_length  # 下一部分的起始位置
        
        print(f"[KV-REUSE] 固定部分: {fixed_length} tokens (位置 0-{fixed_length-1}, 复用)")
        
        # ============ 步骤 2: 获取并校正选中 shots 的 KV ============
        K_shots_all = [[] for _ in range(self.num_layers)]
        V_shots_all = [[] for _ in range(self.num_layers)]
        shots_lengths = []
        
        for shot_id in selected_shot_ids:
            if shot_id not in self.kv_pool.kv_cache_pool:
                continue
            
            # 获取该 shot 的预计算 KV
            cache_entry = self.kv_pool.kv_cache_pool[shot_id]
            K_shot = cache_entry['K_layers']
            V_shot = cache_entry['V_layers']
            shot_len = cache_entry['token_count']
            
            # 对每层进行 RoPE 位置校正
            for layer_idx in range(self.num_layers):
                # 校正 K 的位置（从原始位置 [0, shot_len) 调整到 [current_offset, current_offset+shot_len)）
                K_corrected = self.rope_corrector.apply_rope_offset(
                    K_shot[layer_idx], 
                    offset=current_offset
                )
                K_shots_all[layer_idx].append(K_corrected)
                V_shots_all[layer_idx].append(V_shot[layer_idx])  # V 不需要校正
            
            shots_lengths.append(shot_len)
            current_offset += shot_len
        
        # 拼接所有 shots 的 KV
        total_shots_tokens = sum(shots_lengths)
        print(f"[KV-REUSE] Shots: {len(selected_shot_ids)} 个, 共 {total_shots_tokens} tokens (位置 {fixed_length}-{current_offset-1}, RoPE 校正)")
        
        # ============ 步骤 3: 计算 Query 的 KV（位置从 current_offset 开始） ============
        header = "When you respond, your last line must be exactly of the form '#### <final_answer>'."
        if self.mode == "cot" or self.mode == "paper":
            formatted_query = f"{header}\nProblem: {query_text}\nSolution:"
        else:
            formatted_query = f"{header}\nProblem: {query_text}\nAnswer:"
        
        query_inputs = self.tokenizer(formatted_query, return_tensors="pt")
        query_inputs = {k: v.long() if k == 'input_ids' else v for k, v in query_inputs.items()}
        query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
        query_length = query_inputs["input_ids"].shape[1]
        
        # 设置 query 的 position_ids（从 current_offset 开始）
        query_position_ids = torch.arange(
            current_offset, 
            current_offset + query_length, 
            device=self.device
        ).unsqueeze(0)
        
        print(f"[KV-REUSE] Query: {query_length} tokens (位置 {current_offset}-{current_offset+query_length-1}, 现算)")
        
        # ============ 步骤 4: 拼接所有 KV ============
        K_all_layers = []
        V_all_layers = []
        
        for layer_idx in range(self.num_layers):
            parts_K = []
            parts_V = []
            
            # 1. 固定部分
            if K_fixed_layers[layer_idx].shape[0] > 0:
                parts_K.append(K_fixed_layers[layer_idx])
                parts_V.append(V_fixed_layers[layer_idx])
            
            # 2. Shots（已校正）
            if K_shots_all[layer_idx]:
                parts_K.extend(K_shots_all[layer_idx])
                parts_V.extend(V_shots_all[layer_idx])
            
            # 拼接
            if parts_K:
                K_concat = torch.cat(parts_K, dim=0)
                V_concat = torch.cat(parts_V, dim=0)
                # 🔥 确保 K 和 V 的 dtype 一致（解决 Half/float 不匹配问题）
                target_dtype = V_concat.dtype  # V 是原始 dtype
                K_concat = K_concat.to(dtype=target_dtype)
            else:
                K_concat = torch.zeros((0, self.kv_pool.d_k), device=self.device)
                V_concat = torch.zeros((0, self.kv_pool.d_v), device=self.device)
            
            K_all_layers.append(K_concat)
            V_all_layers.append(V_concat)
        
        # 转换为 past_key_values 格式
        past_kv = self._reshape_kv_to_past_format(K_all_layers, V_all_layers)
        past_length = fixed_length + total_shots_tokens
        
        print(f"[KV-REUSE] 复用的 KV: {past_length} tokens")
        
        # ============ 步骤 5: Forward Query，使用拼接的 KV cache ============
        attention_mask = torch.ones(
            (1, past_length + query_length), 
            dtype=torch.long, 
            device=self.device
        )
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=query_inputs["input_ids"],
                attention_mask=attention_mask,
                position_ids=query_position_ids,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True
            )
            
            full_past_kv = outputs.past_key_values
            full_length = past_length + query_length
        
        # ============ 步骤 6: 生成 ============
        generated_tokens = []
        
        last_token_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        
        current_token = next_token.unsqueeze(0)
        current_past_kv = full_past_kv
        current_length = full_length
        
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        decoded_so_far = ""
        
        with torch.inference_mode():
            for step_idx in range(max_new_tokens - 1):
                pos_ids = torch.tensor([[current_length]], device=self.device)
                attn_mask = torch.ones((1, current_length + 1), dtype=torch.long, device=self.device)
                
                outputs = self.model(
                    input_ids=current_token,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=current_past_kv,
                    use_cache=True,
                    return_dict=True
                )
                
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())
                
                if eos_id is not None and next_token.item() == eos_id:
                    break
                
                try:
                    decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                except Exception:
                    pass
                if "####" in decoded_so_far:
                    break
                
                current_token = next_token.unsqueeze(0)
                current_past_kv = outputs.past_key_values
                current_length += 1
                
                if (step_idx + 1) % 50 == 0:
                    print(f"[KV-REUSE] 已生成 {step_idx + 2} tokens")
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 统计复用率
        reuse_tokens = fixed_length + total_shots_tokens
        total_tokens = reuse_tokens + query_length
        reuse_ratio = reuse_tokens / total_tokens if total_tokens > 0 else 0
        
        print(f"[KV-REUSE] 生成完成, 输出 {len(generated_tokens)} tokens")
        print(f"[KV-REUSE] KV 复用率: {reuse_ratio:.1%} ({reuse_tokens}/{total_tokens} tokens)")
        
        gen_info = {
            'num_shots_used': len(selected_shot_ids),
            'total_kv_layers': self.num_layers,
            'total_kv_tokens': total_tokens,
            'output_tokens': len(generated_tokens),
            'mode': f"{self.mode}_kv_reuse",
            'kv_tokens_per_part': {
                "fixed": fixed_length,
                "shots": total_shots_tokens,
                "query": query_length,
                "reused": reuse_tokens
            },
            'kv_reuse_ratio': reuse_ratio
        }
        
        return response, gen_info
    
    def _generate_with_hybrid_mode(
        self,
        selected_shot_ids: List[int],
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        max_new_tokens: int = 512,
        query_text: str = "",
        **gen_kwargs
    ) -> Tuple[str, Dict]:
        """
        🔥 混合模式生成：固定部分用预计算的 KV cache + 动态部分用文本拼接
        
        优化点：
        1. System Prompt 的 KV cache 可以复用，无需每次重新计算
        2. 动态部分（shots + query）的位置编码从 fixed_length 开始，保证连续性
        """
        # ============ 步骤 1: 获取固定部分的 KV cache （已预计算） ============
        if self.mode == "paper":
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('paper_fixed')
            fixed_text = self.kv_pool.get_fixed_text('paper_fixed')
        else:
            K_fixed_layers, V_fixed_layers = self.kv_pool.get_fixed_kv('system_prompt')
            fixed_text = self.kv_pool.get_fixed_text('system_prompt')
        
        fixed_length = K_fixed_layers[0].shape[0] if K_fixed_layers[0].shape[0] > 0 else 0
        print(f"[HYBRID] 固定部分 KV: {fixed_length} tokens (已预计算，可复用)")
        
        # ============ 步骤 2: 构建动态部分的文本（shots + query） ============
        dynamic_parts = []
        
        # 2.1 Shots 部分
        if selected_shot_ids:
            if self.mode == "paper":
                # Paper 模式：question-only
                shot_parts = ["You will be provided Problems similar to the ones below:"]
                for sid in selected_shot_ids:
                    if sid in self.kv_pool.kv_cache_pool:
                        example = self.kv_pool.kv_cache_pool[sid]['example']
                        q, _ = self.kv_pool.dataset_handler.format_example_cot(example)
                        shot_parts.append(f"Problem: {q}")
                shot_parts.append("—")
                dynamic_parts.append("\n".join(shot_parts))
            else:
                # CoT/IO 模式：完整的 shot
                fmt = "cot" if self.mode == "cot" else "io"
                shots_text = "\n".join([
                    self.kv_pool.format_shot_text(sid, fmt) 
                    for sid in selected_shot_ids 
                    if sid in self.kv_pool.kv_cache_pool
                ])
                dynamic_parts.append(shots_text)
        
        # 2.2 Query 部分
        header = "When you respond, your last line must be exactly of the form '#### <final_answer>'."
        if self.mode == "cot" or self.mode == "paper":
            formatted_query = f"{header}\nProblem: {query_text}\nSolution:"
        else:
            formatted_query = f"{header}\nProblem: {query_text}\nAnswer:"
        dynamic_parts.append(formatted_query)
        
        # 拼接动态部分
        dynamic_text = "\n".join([p for p in dynamic_parts if p]).strip()
        
        # ============ 步骤 3: Tokenize 动态部分 ============
        dynamic_inputs = self.tokenizer(dynamic_text, return_tensors="pt")
        dynamic_inputs = {k: v.long() if k == 'input_ids' else v for k, v in dynamic_inputs.items()}
        dynamic_inputs = {k: v.to(self.device) for k, v in dynamic_inputs.items()}
        
        dynamic_length = dynamic_inputs["input_ids"].shape[1]
        print(f"[HYBRID] 动态部分: {dynamic_length} tokens")
        
        # ============ 步骤 4: 将固定部分 KV 转换为 past_key_values 格式 ============
        if fixed_length > 0:
            fixed_past_kv = self._reshape_kv_to_past_format(K_fixed_layers, V_fixed_layers)
        else:
            fixed_past_kv = None
        
        # ============ 步骤 5: 设置正确的 position_ids（从 fixed_length 开始） ============
        # 关键！这确保动态部分的位置编码与固定部分连续
        position_ids = torch.arange(
            fixed_length, 
            fixed_length + dynamic_length, 
            device=self.device
        ).unsqueeze(0)
        
        # attention_mask 需要覆盖 past + current
        attention_mask = torch.ones(
            (1, fixed_length + dynamic_length), 
            dtype=torch.long, 
            device=self.device
        )
        
        # ============ 步骤 6: Forward 动态部分，使用固定部分的 KV cache ============
        with torch.inference_mode():
            # Prefill 动态部分，复用固定部分的 KV
            outputs = self.model(
                input_ids=dynamic_inputs["input_ids"],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=fixed_past_kv,
                use_cache=True,
                return_dict=True
            )
            
            # 现在 outputs.past_key_values 包含了 固定部分 + 动态部分 的完整 KV
            full_past_kv = outputs.past_key_values
            full_length = fixed_length + dynamic_length
            
            print(f"[HYBRID] 完整 KV: {full_length} tokens (固定 {fixed_length} + 动态 {dynamic_length})")
        
        # ============ 步骤 7: 使用完整的 KV cache 进行生成 ============
        generated_tokens = []
        
        # 获取 动态部分的最后一个 token 作为生成起点
        last_token_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        
        current_token = next_token.unsqueeze(0)
        current_past_kv = full_past_kv
        current_length = full_length
        
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        decoded_so_far = ""
        
        with torch.inference_mode():
            for step_idx in range(max_new_tokens - 1):  # -1 因为已经生成了第一个 token
                # position_ids 为当前位置
                pos_ids = torch.tensor([[current_length]], device=self.device)
                
                # attention_mask 覆盖所有历史 + 当前
                attn_mask = torch.ones((1, current_length + 1), dtype=torch.long, device=self.device)
                
                outputs = self.model(
                    input_ids=current_token,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=current_past_kv,
                    use_cache=True,
                    return_dict=True
                )
                
                # 获取下一个 token
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())
                
                # 检查 EOS
                if eos_id is not None and next_token.item() == eos_id:
                    break
                
                # 检查 #### 标志
                try:
                    decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                except Exception:
                    pass
                if "####" in decoded_so_far:
                    break
                
                # 更新状态
                current_token = next_token.unsqueeze(0)
                current_past_kv = outputs.past_key_values
                current_length += 1
                
                # 进度显示
                if (step_idx + 1) % 50 == 0:
                    print(f"[HYBRID] 已生成 {step_idx + 2} tokens")
        
        # 解码
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"[HYBRID] 生成完成, 输出 {len(generated_tokens)} tokens")
        
        # 生成信息
        gen_info = {
            'num_shots_used': len(selected_shot_ids),
            'total_kv_layers': self.num_layers,
            'total_kv_tokens': full_length,
            'output_tokens': len(generated_tokens),
            'mode': f"{self.mode}_hybrid",
            'kv_tokens_per_part': {
                "fixed": fixed_length, 
                "dynamic": dynamic_length,
                "total": full_length
            },
            'kv_reuse_ratio': fixed_length / full_length if full_length > 0 else 0
        }
        
        return response, gen_info
    
    def _generate_with_kv_concat(
        self,
        selected_shot_ids: List[int],
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        max_new_tokens: int = 512,
        query_text: str = "",
        **gen_kwargs
    ) -> Tuple[str, Dict]:
        """
        ⚠️ 原有的 KV 拼接模式（位置编码可能有问题，仅用于实验对比）
        """
        # 拼装 KV cache
        past_key_values = self.assemble_kv_for_generation(
            selected_shot_ids, prompt_tokens, query_tokens, query_text
        )
        
        # 注意: 由于我们已经有了完整的 KV cache，
        # 生成时不需要再输入完整的 token 序列
        # 只需要一个 dummy input_ids 来触发生成
        
        use_formatted = False
        formatted_ids = None
        if query_text:
            if self.mode == "cot" or self.mode == "paper":
                formatted_query = f"When you respond, your last line must be exactly of the form '#### <final_answer>'.\nProblem: {query_text}\nSolution:"
            else:
                formatted_query = f"Problem: {query_text}\nAnswer:"
            try:
                formatted_ids = self.tokenizer.encode(formatted_query, add_special_tokens=False, return_tensors='pt')[0]
                use_formatted = True
            except Exception:
                use_formatted = False
        if use_formatted and formatted_ids is not None and formatted_ids.numel() > 0:
            start_token = formatted_ids[-1:].long().unsqueeze(0).to(self.device)
        else:
            start_token = query_tokens[-1:].long().unsqueeze(0).to(self.device)
        
        # 计算 past_key_values 的长度（从第一层的 K 获取）
        past_length = past_key_values[0][0].shape[2] if past_key_values else 0
        
        # 创建 attention_mask
        attention_mask = torch.ones((1, past_length + 1), dtype=torch.long, device=self.device)
        
        # 🔥 使用手动循环生成，避免 cache_position 问题
        print(f"[GEN-KV] 开始生成 (max_tokens={max_new_tokens}, seq_len={past_length})")
        generated_tokens = []
        current_token = start_token
        current_past_kv = past_key_values
        
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        decoded_so_far = ""
        with torch.inference_mode():
            for step_idx in range(max_new_tokens):
                # Forward
                pos_ids = torch.arange(past_length, past_length + 1, device=self.device).unsqueeze(0)
                outputs = self.model(
                    input_ids=current_token,
                    attention_mask=attention_mask,
                    past_key_values=current_past_kv,
                    position_ids=pos_ids,
                    use_cache=True,
                    return_dict=True
                )
                
                # 每 50 步显示一次进度
                if step_idx % 50 == 0 and step_idx > 0:
                    current_total_len = past_length + 1 + step_idx
                    print(f"[GEN-KV] 已生成 {step_idx} tokens, 总长度={current_total_len}")
                
                # 获取下一个 token
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # ✅ 修复：添加 token 到生成列表（在检查 EOS 之前）
                generated_tokens.append(next_token.item())
                
                if eos_id is not None and next_token.item() == eos_id:
                    break
                try:
                    decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                except Exception:
                    decoded_so_far = decoded_so_far
                if "####" in decoded_so_far:
                    break
                
                # 准备下一轮
                current_token = next_token.unsqueeze(0)
                current_past_kv = outputs.past_key_values
                
                # 更新 attention_mask
                new_seq_len = past_length + 2 + step_idx
                attention_mask = torch.ones((1, new_seq_len), dtype=torch.long, device=self.device)
                past_length = past_length + 1
            
            # 生成完成提示
            print(f"[GEN-KV] 生成完成, 总共生成 {len(generated_tokens)} tokens")
            # 解码
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        gen_info = {
            'num_shots_used': len(selected_shot_ids),
            'total_kv_layers': self.num_layers,
            'total_kv_tokens': sum(
                self.kv_pool.kv_cache_pool[sid]['token_count'] 
                for sid in selected_shot_ids
            ) + prompt_tokens.shape[0] + query_tokens.shape[0],
            'output_tokens': len(generated_tokens),
            'mode': f"{self.mode}_kv_concat",
            'kv_tokens_per_part': dict(self._last_part_sizes)
        }
        
        return response, gen_info
    
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
