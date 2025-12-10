"""
BM25 检索器
用于构建全局示例池的 BM25 索引并进行检索
"""
import sys
import os
import json
import logging
import pickle
from typing import List, Dict, Tuple
from pathlib import Path

# Merge: 修改路径从 ../baseline 到 ../
# Original: # 添加 baseline 路径以复用代码
# Original: sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../baseline'))
# 添加根路径以定位 util 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "请安装 rank_bm25 库：pip install rank-bm25"
    )

# Merge: 修改导入路径从 dataset_handlers 到 util.dataset_handlers
# Original: from dataset_handlers import get_dataset_handler
from util.dataset_handlers import get_dataset_handler


class BM25Retriever:
    """BM25 检索器，用于从全局示例池中检索相关示例"""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_subset: str = None,
        pool_size: int = 1024,
        use_question_only: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
        seed: int = 42,
        cache_dir: str = "./cache",
        tasks: str = None
    ):
        """
        初始化 BM25 检索器
        
        Args:
            dataset_name: 数据集名称
            dataset_subset: 数据集子集
            pool_size: 全局示例池大小
            use_question_only: 是否只使用问题文本（不包含答案）
            k1: BM25 参数 k1
            b: BM25 参数 b
            seed: 随机种子
            cache_dir: 缓存目录
            tasks: 任务列表（逗号分隔）
        """
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.pool_size = pool_size
        self.use_question_only = use_question_only
        self.k1 = k1
        self.b = b
        self.seed = seed
        self.cache_dir = cache_dir
        self.tasks = tasks
        
        # 获取数据集处理器
        self.dataset_handler = get_dataset_handler(dataset_name, dataset_subset, tasks)
        
        # BM25 索引和示例池
        self.bm25_index = None
        self.pool_examples = None
        self.pool_texts = None
        
        # 创建缓存目录
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"初始化 BM25Retriever: dataset={dataset_name}/{dataset_subset}, "
                    f"pool_size={pool_size}, use_question_only={use_question_only}")
    
    def _get_cache_path(self) -> str:
        """获取缓存文件路径"""
        dataset_key = f"{self.dataset_name}_{self.dataset_subset or 'default'}"
        dataset_key = dataset_key.replace("/", "_").replace(":", "_")
        
        # 如果有 tasks，将其纳入缓存键
        task_suffix = ""
        if self.tasks:
            import hashlib
            task_hash = hashlib.md5(self.tasks.encode()).hexdigest()[:8]
            task_suffix = f"_task{task_hash}"
        
        cache_name = f"bm25_pool_{dataset_key}_size{self.pool_size}_qonly{self.use_question_only}_seed{self.seed}{task_suffix}.pkl"
        return os.path.join(self.cache_dir, cache_name)
    
    def build_index(self, force_rebuild: bool = False):
        """
        构建 BM25 索引
        
        Args:
            force_rebuild: 是否强制重建索引（忽略缓存）
        """
        cache_path = self._get_cache_path()
        
        # 尝试加载缓存
        if not force_rebuild and os.path.exists(cache_path):
            try:
                logging.info(f"从缓存加载 BM25 索引: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                self.bm25_index = cache_data['bm25_index']
                self.pool_examples = cache_data['pool_examples']
                self.pool_texts = cache_data['pool_texts']
                logging.info(f"成功加载缓存，示例池大小: {len(self.pool_examples)}")
                return
            except Exception as e:
                logging.warning(f"加载缓存失败: {e}，将重新构建索引")
        
        # 加载训练集
        logging.info(f"加载训练集: {self.dataset_name}/{self.dataset_subset}")
        train_set, _ = self.dataset_handler.load_and_split(test_size=300, seed=self.seed)
        
        # 选取前 pool_size 个示例
        actual_pool_size = min(self.pool_size, len(train_set))
        self.pool_examples = train_set.select(range(actual_pool_size))
        logging.info(f"示例池大小: {actual_pool_size}")
        
        # 提取检索文本
        logging.info("提取检索文本...")
        self.pool_texts = []
        for i, example in enumerate(self.pool_examples):
            if self.use_question_only:
                # 只使用问题文本
                question, _ = self.dataset_handler.format_example_cot(example)
                text = question
            else:
                # 使用问题+答案
                question, answer = self.dataset_handler.format_example_cot(example)
                text = f"{question} {answer}"
            
            self.pool_texts.append(text)
        
        # 分词（简单按空格分词）
        logging.info("构建 BM25 索引...")
        tokenized_corpus = [text.split() for text in self.pool_texts]
        
        # 创建 BM25 索引
        self.bm25_index = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        # 保存缓存
        logging.info(f"保存 BM25 索引到缓存: {cache_path}")
        cache_data = {
            'bm25_index': self.bm25_index,
            'pool_examples': self.pool_examples,
            'pool_texts': self.pool_texts
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logging.info("BM25 索引构建完成")
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[Dict], List[float]]:
        """
        检索最相关的示例
        
        Args:
            query: 查询文本（通常是问题）
            top_k: 返回前 k 个结果（None 表示返回所有）
        
        Returns:
            (retrieved_examples, scores): 检索到的示例列表和相应的分数
        """
        if self.bm25_index is None:
            raise RuntimeError("BM25 索引未构建，请先调用 build_index()")
        
        # 分词
        tokenized_query = query.split()
        
        # 获取 BM25 分数
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # 排序获取 top-k 索引
        if top_k is None:
            top_k = len(scores)
        
        # 获取排序后的索引
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]
        
        # 获取对应的示例和分数
        retrieved_examples = [self.pool_examples[i] for i in top_indices]
        retrieved_scores = [scores[i] for i in top_indices]
        
        return retrieved_examples, retrieved_scores
    
    def get_pool_info(self) -> Dict:
        """获取示例池信息"""
        if self.pool_examples is None:
            return {"status": "not_built"}
        
        return {
            "status": "ready",
            "pool_size": len(self.pool_examples),
            "dataset": f"{self.dataset_name}/{self.dataset_subset}",
            "use_question_only": self.use_question_only,
            "bm25_params": {"k1": self.k1, "b": self.b}
        }
    
    def save_pool_info(self, output_path: str):
        """保存示例池信息到文件"""
        info = self.get_pool_info()
        
        # 添加示例预览
        if self.pool_examples is not None:
            info['examples_preview'] = []
            for i in range(min(5, len(self.pool_examples))):
                example = self.pool_examples[i]
                q, a = self.dataset_handler.format_example_cot(example)
                info['examples_preview'].append({
                    'index': i,
                    'question': q[:200] + '...' if len(q) > 200 else q,
                    'answer_preview': a[:100] + '...' if len(a) > 100 else a
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        logging.info(f"示例池信息已保存到: {output_path}")


def test_retriever():
    """测试 BM25 检索器"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建检索器
    retriever = BM25Retriever(
        dataset_name="openai/gsm8k",
        dataset_subset="main",
        pool_size=100,
        use_question_only=True
    )
    
    # 构建索引
    retriever.build_index()
    
    # 测试检索
    test_query = "John has 5 apples and buys 3 more. How many apples does he have?"
    examples, scores = retriever.retrieve(test_query, top_k=5)
    
    print(f"\n查询: {test_query}")
    print(f"\n检索到的前 5 个示例:")
    for i, (example, score) in enumerate(zip(examples, scores)):
        q, a = retriever.dataset_handler.format_example_cot(example)
        print(f"\n--- 示例 {i+1} (分数: {score:.4f}) ---")
        print(f"问题: {q[:150]}...")
        print(f"答案: {a[:100]}...")


if __name__ == "__main__":
    test_retriever()
