"""
AdaCache 配置文件
定义所有可配置的参数
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AdaCacheConfig:
    """AdaCache 实验配置"""
    
    # ========== 模式配置 ==========
    mode: str = "cot"  # 模式：cot, io, paper
    
    # ========== 全局示例池配置 ==========
    global_pool_size: int = 1024  # 全局示例池大小
    
    # ========== 窗口与探针配置 ==========
    window_size: int = 4  # 每轮激活的示例数量 n
    entropy_threshold: float = 0.5  # 信息熵阈值 τ
    max_examples: int = 512  # 最大示例数量上限
    max_probe_rounds: int = 128  # 最大探针轮数 (max_examples // window_size)
    
    # ========== Paper 模式专属配置 ==========
    paper_k_full: int = 4  # paper 模式下固定的完整示例（问题-过程-答案）数量
    paper_num_questions: int = 4  # paper 模式下的 question-only shots 数量
    
    # ========== 探针 Prompt 配置 ==========
    probe_question: str = "Based on the above examples, are you confident to answer this question? Please answer Yes or No."
    
    # ========== 模型与数据集配置 ==========
    models: List[str] = field(default_factory=lambda: [
        "/home/models/models/Qwen/Qwen2.5-14B-Instruct"
    ])
    dataset_configs: List[dict] = field(default_factory=lambda: [
        {"name": "openai/gsm8k", "subset": "main"},
        # {"name": "aqua_rat", "subset": None},
        # {"name": "TIGER-Lab/MATH-plus", "subset": "Math-500"},
    ])
    
    # ========== 实验基础配置 ==========
    eval_samples: int = 100  # 评测样本数量
    gen_tokens: int = 4096  # 生成 token 数量（修复：从 512 增加到 4096）
    seed: int = 42  # 随机种子
    tasks: Optional[str] = None  # 任务列表（逗号分隔），用于过滤数据集
    
    # ========== 路径配置 ==========
    model_root: str = "/home/models/models/"
    output_dir: str = "./outputs"
    baseline_dir: str = "../util"  # util 代码路径，用于复用
    
    # ========== 运行配置 ==========
    run_id: str = "adacache_test"
    device: Optional[str] = None  # None 表示自动检测
    use_sharding: bool = True  # 是否使用 NPU 多卡分片
    
    # ========== BM25 配置 ==========
    bm25_k1: float = 1.5  # BM25 参数 k1
    bm25_b: float = 0.75  # BM25 参数 b
    bm25_use_question_only: bool = True  # 是否只使用问题文本进行检索（不包含答案）
    
    # ========== KVCache Token 级选择配置 ==========
    token_top_k_ratio: float = 0.1  # Token 选择比例（选取 top 10% 最相关的 token）
    shot_activation_threshold: float = 0.8  # Shot 激活阈值（80% token 被激活才选中该 shot）
    
    # ========== 自适应选择配置 ==========
    min_shots: int = 1  # 最少选择的 shot 数量
    max_shots: int = 64  # 最多选择的 shot 数量
    target_shots: int = 8  # 目标 shot 数量
    
    # ========== 日志配置 ==========
    verbose: bool = True  # 是否输出详细日志
    save_probe_details: bool = True  # 是否保存每轮探针的详细信息
    

    
    def __post_init__(self):
        """初始化后的验证"""
        # 验证模式
        if self.mode not in ["cot", "io", "paper"]:
            raise ValueError(f"mode 必须是 'cot', 'io' 或 'paper'，当前值: {self.mode}")
        
        # 确保 max_probe_rounds 合理
        if self.max_examples > 0 and self.window_size > 0:
            self.max_probe_rounds = min(
                self.max_probe_rounds,
                self.max_examples // self.window_size
            )
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "probe_details"), exist_ok=True)


def create_config_from_args(args) -> AdaCacheConfig:
    """从命令行参数创建配置"""
    config = AdaCacheConfig()
    
    # 更新配置
    if hasattr(args, 'mode') and args.mode is not None:
        config.mode = args.mode
    if hasattr(args, 'global_pool_size') and args.global_pool_size is not None:
        config.global_pool_size = args.global_pool_size
    if hasattr(args, 'window_size') and args.window_size is not None:
        config.window_size = args.window_size
    if hasattr(args, 'entropy_threshold') and args.entropy_threshold is not None:
        config.entropy_threshold = args.entropy_threshold
    if hasattr(args, 'max_examples') and args.max_examples is not None:
        config.max_examples = args.max_examples
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
    if hasattr(args, 'models') and args.models is not None:
        config.models = [m.strip() for m in args.models.split(',') if m.strip()]
    if hasattr(args, 'datasets') and args.datasets is not None:
        # 支持格式：dataset1:subset1,dataset2:subset2
        config.dataset_configs = []
        for ds in args.datasets.split(','):
            ds = ds.strip()
            if ':' in ds:
                name, subset = ds.split(':', 1)
                config.dataset_configs.append({"name": name.strip(), "subset": subset.strip()})
            else:
                config.dataset_configs.append({"name": ds, "subset": None})
    
    return config
