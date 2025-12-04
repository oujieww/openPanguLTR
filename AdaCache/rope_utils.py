"""
RoPE 位置校正工具

用于在拼接独立计算的 KV cache 时，校正 Key 中嵌入的 RoPE 位置编码。

核心原理:
- RoPE 将位置信息嵌入到 Key 中: K_rotated = K * cos(pos) + rotate_half(K) * sin(pos)
- 独立 prefill 的 shots 各自从 pos=0 开始
- 拼接时需要将每个 shot 的位置偏移到正确的绝对位置

位置校正公式:
  K_corrected = K_old * cos(delta) + rotate_half(K_old) * sin(delta)
  其中 delta = new_pos - old_pos

注意: V (Value) 不需要校正，因为 RoPE 只应用于 Q 和 K
"""
import torch
import math
from typing import List, Tuple, Optional


class RoPECorrector:
    """RoPE 位置校正器"""
    
    def __init__(self, model, device: str = "npu"):
        """
        初始化 RoPE 校正器
        
        Args:
            model: 语言模型（用于获取 RoPE 参数）
            device: 计算设备
        """
        self.device = device
        self.model = model
        
        # 从模型配置中获取 RoPE 参数
        config = model.config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        
        # RoPE 参数
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 32768)
        
        # 预计算 cos/sin 缓存
        self._build_cos_sin_cache()
        
    def _build_cos_sin_cache(self, max_seq_len: int = None):
        """
        构建 cos/sin 缓存（与 Qwen2/Llama 的 RoPE 实现一致）
        """
        if max_seq_len is None:
            max_seq_len = self.max_position_embeddings
        
        # 计算频率（与 transformers 一致）
        dim = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        
        # 位置序列
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        
        # 外积计算 [seq_len, dim/2]
        freqs = torch.outer(positions, inv_freq)
        
        # 扩展为 [seq_len, dim] (重复两次以匹配 head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # 缓存 cos 和 sin（🔥 保存在 CPU 上，避免多 NPU 设备切片问题）
        self.cos_cache = emb.cos()  # [max_seq_len, head_dim]
        self.sin_cache = emb.sin()  # [max_seq_len, head_dim]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        旋转向量的一半（RoPE 标准操作）
        
        Args:
            x: [..., head_dim]
        
        Returns:
            rotated: [..., head_dim]
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rope_offset(
        self,
        K: torch.Tensor,
        offset: int
    ) -> torch.Tensor:
        """
        对 Key 应用 RoPE 位置偏移
        
        原理: 假设 K 是用 pos=[0,1,...,T-1] 计算的
              我们需要将其转换为 pos=[offset, offset+1, ..., offset+T-1]
        
        Args:
            K: Key 张量 [seq_len, d_k] 其中 d_k = num_heads * head_dim
            offset: 位置偏移量
        
        Returns:
            K_corrected: 校正后的 Key [seq_len, d_k]，保持原始 dtype
        """
        if offset == 0:
            return K  # 无需校正
        
        seq_len, d_k = K.shape
        original_dtype = K.dtype  # 🔥 保存原始数据类型
        original_device = K.device  # 🔥 保存原始设备
        
        # 确保 offset 不超过缓存范围
        if offset + seq_len > self.cos_cache.shape[0]:
            self._build_cos_sin_cache(offset + seq_len + 1024)
        
        # Reshape: [seq_len, d_k] -> [seq_len, num_heads, head_dim]
        K_reshaped = K.view(seq_len, self.num_key_value_heads, self.head_dim)
        
        # 获取旧位置和新位置的 cos/sin
        # 🔥 关键修复：先切片，再转换设备和 dtype（避免设备不匹配）
        old_positions = torch.arange(seq_len, device='cpu')  # 在 CPU 上创建索引
        new_positions = old_positions + offset
        
        # 🔥 先在 CPU 上切片，然后转换到目标设备和 dtype
        cos_old = self.cos_cache[old_positions].to(device=original_device, dtype=original_dtype)  # [seq_len, head_dim]
        sin_old = self.sin_cache[old_positions].to(device=original_device, dtype=original_dtype)
        cos_new = self.cos_cache[new_positions].to(device=original_device, dtype=original_dtype)
        sin_new = self.sin_cache[new_positions].to(device=original_device, dtype=original_dtype)
        
        # 扩展维度以匹配 num_heads
        cos_old = cos_old.unsqueeze(1)  # [seq_len, 1, head_dim]
        sin_old = sin_old.unsqueeze(1)
        cos_new = cos_new.unsqueeze(1)
        sin_new = sin_new.unsqueeze(1)
        
        # 反旋转（去除旧位置的 RoPE）
        # RoPE_inv(a): x * cos(a) - rotate_half(x) * sin(a)
        K_unrotated = K_reshaped * cos_old - self._rotate_half(K_reshaped) * sin_old
        
        # 正旋转（应用新位置的 RoPE）
        # RoPE(b): x * cos(b) + rotate_half(x) * sin(b)
        K_corrected = K_unrotated * cos_new + self._rotate_half(K_unrotated) * sin_new
        
        # Reshape 回 [seq_len, d_k]
        K_corrected = K_corrected.view(seq_len, d_k)
        
        # 🔥 确保返回与输入相同的 dtype（关键！防止 float32 和 float16 不匹配）
        return K_corrected.to(dtype=original_dtype)
    
    def correct_kv_positions(
        self,
        K_layers: List[torch.Tensor],
        V_layers: List[torch.Tensor],
        offset: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        对所有层的 KV 进行位置校正
        
        Args:
            K_layers: 所有层的 Key [layer][seq_len, d_k]
            V_layers: 所有层的 Value [layer][seq_len, d_v]
            offset: 位置偏移量
        
        Returns:
            (K_corrected, V_layers): 校正后的 K 和原始 V（V 不需要校正）
        """
        if offset == 0:
            return K_layers, V_layers
        
        K_corrected = []
        for layer_idx, K in enumerate(K_layers):
            K_new = self.apply_rope_offset(K, offset)
            K_corrected.append(K_new)
        
        # V 不需要校正（RoPE 只应用于 Q 和 K）
        return K_corrected, V_layers


def test_rope_corrector():
    """测试 RoPE 校正器"""
    print("Testing RoPE Corrector...")
    
    # 创建假的模型配置
    class FakeConfig:
        hidden_size = 2048
        num_attention_heads = 16
        num_key_value_heads = 4
        rope_theta = 10000.0
        max_position_embeddings = 4096
    
    class FakeModel:
        config = FakeConfig()
    
    corrector = RoPECorrector(FakeModel(), device="cpu")
    
    # 测试数据
    seq_len = 10
    d_k = FakeConfig.num_key_value_heads * (FakeConfig.hidden_size // FakeConfig.num_attention_heads)
    K = torch.randn(seq_len, d_k)
    
    # 测试偏移
    K_offset_0 = corrector.apply_rope_offset(K, offset=0)
    assert torch.allclose(K, K_offset_0), "offset=0 should return identical K"
    
    K_offset_10 = corrector.apply_rope_offset(K, offset=10)
    assert K_offset_10.shape == K.shape, "Shape should be preserved"
    
    print("✓ RoPE Corrector test passed!")


if __name__ == "__main__":
    test_rope_corrector()
