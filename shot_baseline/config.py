"""
Shot Baseline 配置文件
支持三种模式：CoT, IO, Paper
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ShotConfig:
    """Shot Baseline 实验配置"""
    
    # ========== Shot 模式配置 ==========
    mode: str = "cot"  # 模式：cot, io, paper
    num_shots: int = 4  # CoT/IO 模式下的 shot 数量
    
    # ========== Paper 模式专属配置 ==========
    paper_k_full: int = 4  # Paper 模式下的 fullshot 数量
    paper_num_questions: int = 4  # Paper 模式下的 question-only shots 数量
    
    # ========== 模型与数据集配置 ==========
    models: List[str] = None
    dataset_configs: List[dict] = None
    
    # ========== 实验基础配置 ==========
    eval_samples: int = 100  # 评测样本数量
    gen_tokens: int = 512  # 生成 token 数量
    seed: int = 42  # 随机种子
    tasks: Optional[str] = None  # 任务列表（逗号分隔），用于过滤数据集
    
    # ========== 路径配置 ==========
    model_root: str = "/home/models/models/"
    output_dir: str = "./outputs"
    
    # ========== 运行配置 ==========
    run_id: str = "shot_test"
    device: Optional[str] = None  # None 表示自动检测
    use_sharding: bool = True  # 是否使用 NPU 多卡分片
    
    # ========== 日志配置 ==========
    verbose: bool = True  # 是否输出详细日志
    
    def __post_init__(self):
        """初始化后的验证"""
        # 验证模式
        if self.mode not in ["cot", "io", "paper"]:
            raise ValueError(f"mode 必须是 'cot', 'io' 或 'paper'，当前值: {self.mode}")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)


def create_config_from_args(args) -> ShotConfig:
    """从命令行参数创建配置"""
    config = ShotConfig()
    
    # 更新配置
    if hasattr(args, 'mode') and args.mode is not None:
        config.mode = args.mode
    if hasattr(args, 'num_shots') and args.num_shots is not None:
        config.num_shots = args.num_shots
    if hasattr(args, 'paper_k_full') and args.paper_k_full is not None:
        config.paper_k_full = args.paper_k_full
    if hasattr(args, 'paper_num_questions') and args.paper_num_questions is not None:
        config.paper_num_questions = args.paper_num_questions
    if hasattr(args, 'eval_samples') and args.eval_samples is not None:
        config.eval_samples = args.eval_samples
    if hasattr(args, 'gen_tokens') and args.gen_tokens is not None:
        config.gen_tokens = args.gen_tokens
    if hasattr(args, 'seed') and args.seed is not None:
        config.seed = args.seed
    if hasattr(args, 'tasks') and args.tasks is not None:
        config.tasks = args.tasks
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        config.output_dir = args.output_dir
    if hasattr(args, 'run_id') and args.run_id is not None:
        config.run_id = args.run_id
    if hasattr(args, 'model_root') and args.model_root is not None:
        config.model_root = args.model_root
    if hasattr(args, 'verbose') and args.verbose is not None:
        config.verbose = args.verbose
    if hasattr(args, 'device') and args.device is not None:
        config.device = args.device
    if hasattr(args, 'use_sharding') and args.use_sharding is not None:
        config.use_sharding = args.use_sharding
    
    return config
