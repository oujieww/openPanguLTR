"""
KV Cache Pool Manager (华为 910B NPU 优化版本)
负责 1024-shot 的离线 prefilling、KV cache 构建、存储和检索

设计要点:
1. KV 多层设计: 保存所有层的KV，K_i: List[[T_i, d_k]], V_i: List[[T_i, d_v]]
2. 设备兼容: 支持 NPU/CUDA/CPU，所有设备操作通过参数控制
3. 存储策略: 计算并存储在 NPU 内存中，支持磁盘持久化
4. 索引管理: 维护 shot_id -> KV 张量映射
"""
import sys
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm

# Merge: 修改路径从 ../baseline 到 ../util
# Original: # 添加 baseline 路径
# Original: sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../baseline'))
# 添加根路径以定位 util 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from util.dataset_handlers import BaseDatasetHandler


class KVPoolManager:
    """KV Cache 池管理器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        dataset_handler: BaseDatasetHandler,
        pool_size: int = 1024,
        cache_dir: str = "./kv_pool",
        device: str = "npu",
        mode: str = "cot"  # 添加模式参数
    ):
        """
        初始化 KV Pool 管理器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            dataset_handler: 数据集处理器
            pool_size: 示例池大小 (默认 1024)
            cache_dir: KV cache 存储目录
            device: 计算设备 ('npu', 'cuda', 'cpu')
            mode: 模式 ('cot', 'io', 'paper')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_handler = dataset_handler
        self.pool_size = pool_size
        self.cache_dir = cache_dir
        self.device = device
        self.mode = mode
        
        # 确保缓存目录存在
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # KV 池 - 存储在 NPU 内存中
        # 保存原始示例以支持多种格式
        self.kv_cache_pool = {}  # {shot_id: {'K_layers': List[Tensor], 'V_layers': List[Tensor], 'example': Dict, 'token_count': int}}
        
        # ✅ 新增：固定部分的 KV 缓存（System Prompt、Paper 固定部分等）
        self.fixed_kv_cache = {}  # {'system_prompt': KV, 'paper_fixed': KV, etc.}
        
        # 模型配置
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_key_value_heads = getattr(model.config, 'num_key_value_heads', self.num_heads)
        self.head_dim = model.config.hidden_size // self.num_heads
        self.d_k = self.num_key_value_heads * self.head_dim
        self.d_v = self.num_key_value_heads * self.head_dim
        
        logging.info(f"初始化 KVPoolManager: pool_size={pool_size}, device={device}, mode={mode}")
    
    def _extract_kv_from_model_output(self, outputs) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        从模型输出中提取所有层的 KV
        
        保存所有层的 KV:
        - K_layers: List[K_layer], 每个 K_layer: [T, d_k]
        - V_layers: List[V_layer], 每个 V_layer: [T, d_v]
        
        Args:
            outputs: 模型输出 (包含 past_key_values)
        
        Returns:
            (K_layers, V_layers): 所有层的 Key 和 Value 张量列表
        """
        past_kv = outputs.past_key_values
        
        # past_key_values 格式: tuple of layers
        # 每层: (key, value)，形状 [batch, num_heads, seq_len, head_dim]
        
        K_layers = []
        V_layers = []
        
        for layer_idx in range(len(past_kv)):
            layer_k = past_kv[layer_idx][0]  # [batch, num_heads, seq_len, head_dim]
            layer_v = past_kv[layer_idx][1]  # [batch, num_heads, seq_len, head_dim]
            
            # 去掉 batch 维度 (假设 batch=1)
            layer_k = layer_k.squeeze(0)  # [num_heads, seq_len, head_dim]
            layer_v = layer_v.squeeze(0)  # [num_heads, seq_len, head_dim]
            
            # 规约: 将多头合并 (flatten heads)
            # [num_heads, seq_len, head_dim] -> [seq_len, num_heads * head_dim]
            num_heads, seq_len, head_dim = layer_k.shape
            
            K = layer_k.permute(1, 0, 2).reshape(seq_len, num_heads * head_dim)  # [T, d_k]
            V = layer_v.permute(1, 0, 2).reshape(seq_len, num_heads * head_dim)  # [T, d_v]
            
            K_layers.append(K)
            V_layers.append(V)
        
        return K_layers, V_layers
    
    def _save_kv_to_memory(self, shot_id: int, K_layers: List[torch.Tensor], V_layers: List[torch.Tensor], example: Dict):
        """
        将 KV 保存到 NPU 内存
        
        Args:
            shot_id: Shot ID
            K_layers: 所有层的 Key 张量列表
            V_layers: 所有层的 Value 张量列表
            example: 原始示例（用于后续格式化）
        """
        # 存储在 NPU 内存中
        self.kv_cache_pool[shot_id] = {
            'K_layers': K_layers,  # 保持在 device 上
            'V_layers': V_layers,
            'token_count': K_layers[0].shape[0],
            'example': example  # 保存原始示例
        }
    
    def _save_kv_to_disk(self, shot_id: int, K_layers: List[torch.Tensor], V_layers: List[torch.Tensor], text: str):
        """
        将 KV 持久化到磁盘（可选）
        
        Args:
            shot_id: Shot ID
            K_layers: 所有层的 Key 张量列表
            V_layers: 所有层的 Value 张量列表
            text: Shot 文本内容
        """
        # 保存每一层
        for layer_idx in range(len(K_layers)):
            K_path = os.path.join(self.cache_dir, f"shot_{shot_id}_layer_{layer_idx}_K.pt")
            V_path = os.path.join(self.cache_dir, f"shot_{shot_id}_layer_{layer_idx}_V.pt")
            
            torch.save(K_layers[layer_idx].cpu(), K_path)
            torch.save(V_layers[layer_idx].cpu(), V_path)
    
    def _load_kv_from_memory(self, shot_id: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        从 NPU 内存加载 KV
        
        Args:
            shot_id: Shot ID
        
        Returns:
            (K_layers, V_layers): 所有层的 Key 和 Value 张量列表
        """
        if shot_id not in self.kv_cache_pool:
            raise ValueError(f"Shot {shot_id} 不存在于 KV 池中")
        
        cache_entry = self.kv_cache_pool[shot_id]
        return cache_entry['K_layers'], cache_entry['V_layers']

    def get_fixed_text(self, key: str) -> str:
        entry = self.fixed_kv_cache.get(key)
        if not entry:
            return ""
        return entry.get('text', "")
    
    def build_kv_pool(self, examples, force_rebuild: bool = False, system_prompt: str = "", paper_config: dict = None):
        """
        离线 prefilling: 为 1024 个 shot 构建 KV cache 池
        同时缓存固定部分（System Prompt、Paper 固定部分）
        
        ⚠️ 重要：只缓存训练集示例，绞不包含测试集问题，避免数据污染
        
        Args:
            examples: 示例列表（必须是训练集，不能包含测试集）
            force_rebuild: 是否强制重建
            system_prompt: System prompt 文本
            paper_config: Paper 模式配置 {'fullshot_ids': [0,1,2,3], 'num_questions': 4}
        """
        # 注意：KV池存储在NPU内存中，不支持跨会话缓存
        if not force_rebuild:
            logging.info("KV池存储在NPU内存，需要每次重新构建")
        
        # ============ 步骤 1: 缓存 System Prompt ============
        if system_prompt:
            logging.info(f"缓存 System Prompt (长度: {len(system_prompt)} 字符)...")
            K_sys, V_sys = self.get_kv_for_text(system_prompt)
            self.fixed_kv_cache['system_prompt'] = {
                'K_layers': K_sys,
                'V_layers': V_sys,
                'token_count': K_sys[0].shape[0],
                'text': system_prompt
            }
            logging.info(f"✅ System Prompt KV 已缓存，tokens: {K_sys[0].shape[0]}")
        
        # ============ 步骤 2: 缓存 Paper 模式的固定部分 ============
        if self.mode == "paper" and paper_config:
            logging.info("缓存 Paper 模式固定部分（引导语 + fullshots）...")
            fullshot_ids = paper_config.get('fullshot_ids', [])
            
            # 构建固定部分文本
            parts = []
            if system_prompt:
                parts.append(system_prompt)
            
            parts.append("Now, I am going to give you a series of demonstrations of Problems and Solutions to specify the output format.")
            parts.append("When you respond, think step by step, but your last line must be exactly of the form '#### <final_answer>'.")
            
            # 添加 fullshots
            for sid in fullshot_ids:
                if sid < len(examples):
                    ex = examples[sid]
                    q, cot = self.dataset_handler.format_example_cot(ex)
                    parts.append(f"Problem: {q}\nSolution: {cot}")
            
            parts.append("—")  # 分隔符
            
            fixed_text = "\n".join(parts)
            K_fixed, V_fixed = self.get_kv_for_text(fixed_text)
            self.fixed_kv_cache['paper_fixed'] = {
                'K_layers': K_fixed,
                'V_layers': V_fixed,
                'token_count': K_fixed[0].shape[0],
                'text': fixed_text
            }
            logging.info(f"✅ Paper 固定部分 KV 已缓存，tokens: {K_fixed[0].shape[0]}")
        
        # ============ 步骤 3: 离线 prefilling 所有 shots ============
        logging.info(f"开始离线 prefilling {len(examples)} 个 shot...")
        
        with torch.inference_mode():
            for shot_id, example in enumerate(tqdm(examples, desc="Prefilling KV Pool")):
                try:
                    # 格式化 shot 文本（根据模式使用一致的格式）
                    q, a = self.dataset_handler.format_example_cot(example)
                    
                    # ✅ 使用与在线查询一致的格式
                    if self.mode == "cot":
                        shot_text = f"Problem: {q}\nSolution: {a}"
                    elif self.mode == "io":
                        answer = self.dataset_handler.extract_gold_answer(example)
                        shot_text = f"Problem: {q}\nAnswer: {answer}"
                    else:  # 默认使用 CoT 格式
                        shot_text = f"Problem: {q}\nSolution: {a}"
                    
                    # Tokenize
                    inputs = self.tokenizer(shot_text, return_tensors="pt")
                    # 确保 input_ids 是整数类型（NPU要求）
                    inputs = {k: v.long() if k == 'input_ids' else v for k, v in inputs.items()}
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 模型前向 (prefill)
                    outputs = self.model(
                        **inputs,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    # 提取所有层的 KV
                    K_layers, V_layers = self._extract_kv_from_model_output(outputs)
                    
                    # 保存到 NPU 内存（保存原始 example）
                    self._save_kv_to_memory(shot_id, K_layers, V_layers, example)
                    
                    # 可选：持久化到磁盘（用于跨会话缓存）
                    # self._save_kv_to_disk(shot_id, K_layers, V_layers, shot_text)
                    
                except Exception as e:
                    logging.warning(f"Shot {shot_id} prefilling 失败: {e}")
                    continue
        
        logging.info(f"KV 池构建完成，共 {len(self.kv_cache_pool)} 个 shot")
        logging.info(f"KV 池占用 NPU 内存，总 tokens: {sum(v['token_count'] for v in self.kv_cache_pool.values())}")
    
    def get_kv_for_shots(self, shot_ids: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        获取指定 shot 的 KV (拼接所有层)
        
        Args:
            shot_ids: Shot ID 列表
        
        Returns:
            (K_all_layers, V_all_layers): 每层拼接后的 KV 列表
        """
        # 为每一层准备拼接列表
        K_all_layers = [[] for _ in range(self.num_layers)]
        V_all_layers = [[] for _ in range(self.num_layers)]
        
        for shot_id in shot_ids:
            K_layers, V_layers = self._load_kv_from_memory(shot_id)
            
            for layer_idx in range(self.num_layers):
                K_all_layers[layer_idx].append(K_layers[layer_idx])
                V_all_layers[layer_idx].append(V_layers[layer_idx])
        
        # 在 sequence 维度拼接每一层
        K_all_layers = [torch.cat(k_list, dim=0) for k_list in K_all_layers]  # List[[T_total, d_k]]
        V_all_layers = [torch.cat(v_list, dim=0) for v_list in V_all_layers]  # List[[T_total, d_v]]
        
        return K_all_layers, V_all_layers
    
    def get_pool_info(self) -> Dict:
        """获取 KV 池信息"""
        if not self.kv_cache_pool:
            return {"status": "not_built"}
        
        total_tokens = sum(info['token_count'] for info in self.kv_cache_pool.values())
        fixed_tokens = sum(info['token_count'] for info in self.fixed_kv_cache.values())
        
        return {
            "status": "ready",
            "num_shots": len(self.kv_cache_pool),
            "num_layers": self.num_layers,
            "total_tokens": total_tokens,
            "avg_tokens_per_shot": total_tokens / len(self.kv_cache_pool),
            "fixed_tokens": fixed_tokens,
            "fixed_parts": list(self.fixed_kv_cache.keys()),
            "d_k": self.d_k,
            "d_v": self.d_v,
            "storage": "NPU_memory",
            "mode": self.mode
        }
    
    def get_fixed_kv(self, key: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        获取固定部分的 KV（System Prompt、Paper 固定部分等）
        
        Args:
            key: 缓存键名 ('system_prompt', 'paper_fixed')
        
        Returns:
            (K_layers, V_layers): 所有层的 KV 张量列表
        """
        if key not in self.fixed_kv_cache:
            # 返回空 KV
            empty_K = [torch.zeros((0, self.d_k), device=self.device) for _ in range(self.num_layers)]
            empty_V = [torch.zeros((0, self.d_v), device=self.device) for _ in range(self.num_layers)]
            return empty_K, empty_V
        
        cache_entry = self.fixed_kv_cache[key]
        return cache_entry['K_layers'], cache_entry['V_layers']
    
    def get_kv_for_text(self, text: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        为任意文本计算 KV cache
        用于构建 prompt、query 等非-shot 内容的 KV
        
        Args:
            text: 输入文本
        
        Returns:
            (K_layers, V_layers): 所有层的 KV 张量列表
        """
        with torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt")
            # 确保 input_ids 是整数类型（NPU要求）
            inputs = {k: v.long() if k == 'input_ids' else v for k, v in inputs.items()}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True
            )
            
            K_layers, V_layers = self._extract_kv_from_model_output(outputs)
            return K_layers, V_layers
    
    def format_shot_text(self, shot_id: int, format_type: str) -> str:
        """
        根据指定格式返回 shot 的文本
        
        Args:
            shot_id: Shot ID
            format_type: 格式类型 ('cot', 'io', 'question_only')
        
        Returns:
            格式化后的文本
        """
        if shot_id not in self.kv_cache_pool:
            raise ValueError(f"Shot {shot_id} 不存在于 KV 池中")
        
        example = self.kv_cache_pool[shot_id]['example']
        q, cot = self.dataset_handler.format_example_cot(example)
        
        if format_type == "cot":
            return f"Problem: {q}\nSolution: {cot}"
        elif format_type == "io":
            answer = self.dataset_handler.extract_gold_answer(example)
            return f"Problem: {q}\nAnswer: {answer}"
        elif format_type == "question_only":
            return f"Problem: {q}"
        else:
            raise ValueError(f"未知格式类型: {format_type}")
    
    def get_kv_for_shots_formatted(self, shot_ids: List[int], format_type: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        获取指定 shot 的 KV，按指定格式重新计算
        注意：这会重新计算 KV，不使用预计算的 cache
        
        Args:
            shot_ids: Shot ID 列表
            format_type: 格式类型 ('cot', 'io', 'question_only')
        
        Returns:
            (K_all_layers, V_all_layers): 每层拼接后的 KV 列表
        """
        # 为每一层准备拼接列表
        K_all_layers = [[] for _ in range(self.num_layers)]
        V_all_layers = [[] for _ in range(self.num_layers)]
        
        for shot_id in shot_ids:
            # 格式化 shot 文本
            shot_text = self.format_shot_text(shot_id, format_type)
            
            # 计算 KV
            K_layers, V_layers = self.get_kv_for_text(shot_text)
            
            for layer_idx in range(self.num_layers):
                K_all_layers[layer_idx].append(K_layers[layer_idx])
                V_all_layers[layer_idx].append(V_layers[layer_idx])
        
        # 在 sequence 维度拼接每一层
        K_all_layers = [torch.cat(k_list, dim=0) for k_list in K_all_layers]
        V_all_layers = [torch.cat(v_list, dim=0) for v_list in V_all_layers]
        
        return K_all_layers, V_all_layers
