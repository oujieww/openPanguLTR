"""
Shot 排序器
基于 query 的平均 Q 向量，对 KV 池中的 shots 进行打分和排序

流程:
1. Token 级打分: 对每个 shot 的每个 token 计算与 q̄ 的相似度
2. Shot 级聚合: 对每个 shot 的所有 token 得分取平均
3. Shot 排序: 按照平均得分从大到小排序
"""
import sys
import os
import torch
import logging
from typing import List, Tuple
from tqdm.auto import tqdm

# Merge: 修改路径从 ../baseline 到当前目录
# Original: # 添加 baseline 路径
# Original: sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../baseline'))
# 添加当前目录到路径以引用同级模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from kv_pool_manager import KVPoolManager
from query_encoder import compute_cosine_similarity


class ShotRanker:
    """Shot 排序器"""
    
    def __init__(
        self,
        kv_pool_manager: KVPoolManager,
        verbose: bool = True
    ):
        """
        初始化 Shot 排序器
        
        Args:
            kv_pool_manager: KV 池管理器
            verbose: 是否输出详细日志
        """
        self.kv_pool = kv_pool_manager
        self.verbose = verbose
        
        logging.info("初始化 ShotRanker")
    
    def rank_shots(self, query_repr: torch.Tensor) -> List[int]:
        """
        对所有 shots 进行排序
        
        Args:
            query_repr: Query 的平均 Q 向量 [d_q]
        
        Returns:
            ranked_shots: 排序后的 shot ID 列表 (从高到低)
        """
        shot_scores = []
        
        num_shots = len(self.kv_pool.kv_cache_pool)
        
        if self.verbose:
            logging.info(f"开始对 {num_shots} 个 shot 进行排序...")
        
        # 计算每个 shot 的得分
        iterator = tqdm(range(num_shots), desc="Shot Ranking") if self.verbose else range(num_shots)
        
        for shot_id in iterator:
            try:
                # 加载该 shot 的 KV (仅需要最后一层的 K 用于排序)
                K_layers, _ = self.kv_pool._load_kv_from_memory(shot_id)
                K_last = K_layers[-1]  # 使用最后一层 [T_i, d_k]
                
                # 确保 K_last 在正确的设备上
                K_last = K_last.to(query_repr.device)
                
                # Token 级打分: 计算 query_repr 与每个 token 的相似度
                token_similarities = compute_cosine_similarity(query_repr, K_last)  # [T_i]
                
                # Shot 级聚合: 取平均
                shot_score = token_similarities.mean().item()
                
                shot_scores.append((shot_id, shot_score))
                
            except Exception as e:
                logging.warning(f"Shot {shot_id} 排序失败: {e}")
                shot_scores.append((shot_id, -1.0))  # 失败的 shot 给最低分
        
        # 按照得分排序 (从高到低)
        shot_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_shots = [shot_id for shot_id, _ in shot_scores]
        
        if self.verbose:
            logging.info(f"Shot 排序完成，前 10 名得分:")
            for i in range(min(10, len(shot_scores))):
                shot_id, score = shot_scores[i]
                logging.info(f"  #{i+1}: Shot {shot_id}, Score: {score:.4f}")
        
        return ranked_shots
