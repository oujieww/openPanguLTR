"""
Query 编码器
提取查询的平均 Q 向量表征

设计要点:
1. 输入: [prompt_tokens, query_tokens]
2. 输出: 平均 Q 向量 q̄ = 1/T * Σ Q[t]
3. Q 向量使用最后一层的 hidden states，与 K 维度对齐
4. 使用投影层将 hidden_dim 映射到 d_k 以匹配 Key 维度
"""
import sys
import os
import torch
import logging
from typing import Tuple

# 添加 util 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'util'))


class QueryEncoder:
    """查询编码器，提取平均 Q 向量"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "npu"
    ):
        """
        初始化查询编码器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 模型配置
        self.num_heads = model.config.num_attention_heads
        self.num_key_value_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.hidden_dim = model.config.hidden_size
        # 使用 num_key_value_heads 计算 d_k，以匹配 KV cache 的维度
        self.d_k = self.num_key_value_heads * self.head_dim  # Key 维度
        
        # 创建投影层：hidden_dim -> d_k
        # 如果维度相同则不需要投影
        if self.hidden_dim != self.d_k:
            self.projection = torch.nn.Linear(self.hidden_dim, self.d_k).to(device)
            logging.info(f"创建投影层: hidden_dim={self.hidden_dim} -> d_k={self.d_k}")
        else:
            self.projection = None
            logging.info(f"hidden_dim == d_k = {self.d_k}，不需要投影")
        
        logging.info(f"初始化 QueryEncoder: device={device}, d_k={self.d_k}")
    
    def _extract_q_from_hidden_states(self, outputs) -> torch.Tensor:
        """
        从模型隐状态提取 Q 表征并投影到 d_k 维度
        
        流程:
        1. 获取最后一层的隐状态 [T, hidden_dim]
        2. 投影到 d_k 维度 [T, d_k]
        
        Args:
            outputs: 模型输出
        
        Returns:
            Q_all: Query 向量 [T, d_k]
        """
        # 获取最后一层的隐状态
        # hidden_states: [batch, seq_len, hidden_dim]
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]  # 最后一层
            last_hidden = last_hidden.squeeze(0)  # 去掉 batch: [T, hidden_dim]
        else:
            # 降级: 使用 last_hidden_state
            last_hidden = outputs.last_hidden_state.squeeze(0)  # [T, hidden_dim]
        
        # 投影到 d_k 维度
        if self.projection is not None:
            Q_all = self.projection(last_hidden)  # [T, d_k]
        else:
            Q_all = last_hidden  # [T, d_k] (已经匹配)
        
        return Q_all
    
    def encode_query(
        self,
        prompt_tokens: torch.Tensor,
        query_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        编码查询，返回平均 Q 向量
        
        Args:
            prompt_tokens: Prompt token IDs
            query_tokens: Query token IDs
        
        Returns:
            query_repr: 平均 Q 向量 [d_q]
        """
        # 拼接 prompt 和 query
        input_ids = torch.cat([prompt_tokens, query_tokens], dim=-1)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, T]
        
        # 确保是整数类型（NPU要求）
        input_ids = input_ids.long().to(self.device)
        
        with torch.inference_mode():
            # 运行模型，获取隐状态
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 提取 Q 向量
            Q_all = self._extract_q_from_hidden_states(outputs)  # [T, d_k]
            
            # 平均所有 token 的 Q
            query_repr = Q_all.mean(dim=0)  # [d_q]
        
        return query_repr
    
    def encode_query_from_text(
        self,
        prompt_text: str,
        query_text: str
    ) -> torch.Tensor:
        """
        从文本编码查询
        
        Args:
            prompt_text: Prompt 文本
            query_text: Query 文本
        
        Returns:
            query_repr: 平均 Q 向量 [d_q]
        """
        # Tokenize
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        query_tokens = self.tokenizer(query_text, return_tensors="pt").input_ids
        
        return self.encode_query(prompt_tokens, query_tokens)


def compute_cosine_similarity(q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    计算 query 与 Key 矩阵的余弦相似度
    
    Args:
        q: Query 向量 [d_q]
        K: Key 矩阵 [T, d_k]，假设 d_q == d_k
    
    Returns:
        similarities: 相似度向量 [T]
    """
    # 归一化
    q_norm = q / (q.norm() + 1e-8)
    K_norm = K / (K.norm(dim=1, keepdim=True) + 1e-8)
    
    # 余弦相似度
    similarities = torch.matmul(K_norm, q_norm)  # [T]
    
    return similarities
