# -*- coding: utf-8 -*-
"""
数据集处理器：统一不同数据集的加载、划分和格式化
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
# from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names, concatenate_datasets
try:
    from modelscope.msdatasets import MsDataset
except Exception:
    MsDataset = None

class BaseDatasetHandler(ABC):
    """数据集处理器基类"""

    def __init__(self, dataset_name: str, subset: str = None, cache_dir: str = "./dataset_splits"):
        self.dataset_name = dataset_name
        self.subset = subset
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_and_split(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        """加载数据集并返回 (train, test)"""
        pass

    @abstractmethod
    def format_example_io(self, example: Dict) -> Tuple[str, str]:
        """格式化为 I/O 模式的 (question, answer)"""
        pass

    @abstractmethod
    def format_example_cot(self, example: Dict) -> Tuple[str, str]:
        """格式化为 COT 模式的 (question, answer_with_reasoning)"""
        pass

    @abstractmethod
    def extract_gold_answer(self, example: Dict) -> str:
        """提取标准答案"""
        pass

    @abstractmethod
    def extract_prediction(self, model_output: str) -> str:
        """从模型输出中提取预测答案"""
        pass

    def _get_split_cache_path(self, test_size: int, seed: int) -> str:
        """生成数据集划分的缓存路径"""
        cache_id = f"{self.dataset_name}_{self.subset or 'default'}_{test_size}_{seed}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"split_{cache_hash}.json")

    def _save_split_indices(self, train_indices: List[int], test_indices: List[int], path: str):
        """保存数据集划分索引"""
        with open(path, 'w') as f:
            json.dump({
                'train_indices': train_indices,
                'test_indices': test_indices
            }, f)

    def _load_split_indices(self, path: str) -> Optional[Tuple[List[int], List[int]]]:
        """加载数据集划分索引"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return data['train_indices'], data['test_indices']
        return None

    @staticmethod
    def get_available_subsets(dataset_name: str) -> List[str]:
        """获取数据集的所有可用子集"""
        try:
            subsets = get_dataset_config_names(dataset_name)
            return subsets
        except Exception as e:
            print(f"Error getting subsets for {dataset_name}: {e}")
            return []


class GSM8KHandler(BaseDatasetHandler):
    """GSM8K 数据集处理器"""

    def load_and_split(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        # GSM8K 默认使用 main subset
        subset = self.subset or "main"
        dataset = load_dataset(self.dataset_name, subset)

        # GSM8K 已有 train/test 划分
        train = dataset["train"]
        test = dataset["test"]
        if test_size < len(test):
            # 保证相同的seed得到相同的测试集
            np.random.seed(seed)
            test_indices = np.random.choice(len(test), test_size, replace=False).tolist()
            test_indices.sort()  # 排序确保顺序一致
            test = test.select(test_indices)
        return train, test

    def format_example_io(self, example: Dict) -> Tuple[str, str]:
        q = example["question"].strip()
        a_raw = example["answer"].strip()
        final_answer = a_raw.split("####")[-1].strip()
        return q, f"#### {final_answer}"

    def format_example_cot(self, example: Dict) -> Tuple[str, str]:
        q = example["question"].strip()
        a = example["answer"].strip()
        return q, a

    def extract_gold_answer(self, example: Dict) -> str:
        return example["answer"].split("####")[-1].strip()

    def extract_prediction(self, model_output: str) -> str:
        """
        从模型输出中提取预测答案
        
        支持多种格式:
        1. #### 后跟数字: "#### 8" 或 "####8"
        2. 数字后跟 ####: "$57.00.\n####" -> 提取 $57.00 之前的最后一个数字
        3. 带逗号的数字: "$8,400" -> 8400
        4. 带空格的数字: "$9 500" -> 9500
        """
        import re
        text = model_output.strip()
        
        # 辅助函数：清理数字字符串（移除逗号和空格）
        def clean_number(num_str: str) -> str:
            return re.sub(r'[,\s]', '', num_str)
        
        # 辅助函数：提取数字（支持带逗号和空格的格式）
        def extract_number(s: str) -> str:
            # 支持千分位逗号: 1,234,567.89
            m = re.search(r'(-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)', s)
            if m:
                return clean_number(m.group(1))
            # 普通数字
            m = re.search(r'(-?\d+(?:\.\d+)?)', s)
            if m:
                return m.group(1)
            return None
        
        # 🔥 辅助函数：提取最后一个数字（用于从 #### 前提取答案）
        def extract_last_number(s: str) -> str:
            # 先尝试匹配带千分位的大数字
            nums = re.findall(r'-?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?', s)
            if nums:
                return clean_number(nums[-1])
            # 再尝试普通数字
            nums = re.findall(r'-?\d+(?:\.\d+)?', s)
            if nums:
                return nums[-1]
            return None
        
        # 策略 1: 检查 #### 后是否有数字
        if "####" in text:
            parts = text.split("####")
            if len(parts) > 1:
                after_hash = parts[1].strip()
                # 尝试从 #### 后提取数字
                lines = after_hash.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    num = extract_number(line)
                    if num:
                        return num
                    break
                
                # 如果 #### 后没有数字，可能是 "答案\n####" 格式
                # 🔥 修复：从 #### 前面的文本中提取最后一个数字
                before_hash = parts[0].strip()
                if before_hash:
                    num = extract_last_number(before_hash)
                    if num:
                        return num
        
        # 策略 2: 提取最后一个数字（支持带逗号格式）
        nums = re.findall(r'-?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?', text)
        if nums:
            return clean_number(nums[-1])
        
        nums = re.findall(r'-?\d+(?:\.\d+)?', text)
        if nums:
            return nums[-1]
        
        return text


class AquaRatHandler(BaseDatasetHandler):
    """AQUA-RAT 数据集处理器"""

    def load_and_split(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        # AQUA-RAT 默认使用 raw subset
        subset = self.subset or "raw"
        dataset = load_dataset(self.dataset_name, subset)

        # AQUA-RAT 有 train/validation/test，我们使用 train 和 test
        train = dataset["train"]
        test = dataset["test"]

        if test_size < len(test):
            np.random.seed(seed)
            test_indices = np.random.choice(len(test), test_size, replace=False).tolist()
            test_indices.sort()
            test = test.select(test_indices)

        return train, test

    def format_example_io(self, example: Dict) -> Tuple[str, str]:
        q = example["question"].strip()
        options = " ".join(example["options"])
        correct = example["correct"].strip()
        return f"{q}\nOptions: {options}", f"The answer is {correct}"

    def format_example_cot(self, example: Dict) -> Tuple[str, str]:
        q = example["question"].strip()
        options = " ".join(example["options"])
        rationale = example["rationale"].strip()
        correct = example["correct"].strip()
        return f"{q}\nOptions: {options}", f"{rationale}\nThe answer is {correct}"

    def extract_gold_answer(self, example: Dict) -> str:
        return example["correct"].strip()

    def extract_prediction(self, model_output: str) -> str:
        # 查找模式 "The answer is X" 或直接的选项
        import re

        # 首先尝试找 GSM8K 风格的 #### X 格式（用于 paper 模式）
        gsm_match = re.search(r'####\s*([A-E])', model_output, re.IGNORECASE)
        if gsm_match:
            return gsm_match.group(1).upper()

        # 查找 "answer is" 模式，允许后面有标点符号
        answer_pattern = r"(?:the\s+)?answer\s+is\s+([A-E])(?:[.,;:]|\s|$)"
        match = re.search(answer_pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 查找带句号的独立选项（如 "D." 在句尾）
        option_with_punct_pattern = r"\b([A-E])\.(?:\s|$)"
        match = re.search(option_with_punct_pattern, model_output)
        if match:
            return match.group(1).upper()

        # 最后查找独立的选项字母（不带标点）
        option_pattern = r"\b([A-E])\b"
        matches = re.findall(option_pattern, model_output)
        if matches:
            return matches[-1].upper()  # 返回最后一个匹配的选项

        return model_output.strip()

class Math500Handler(BaseDatasetHandler):
    """Math-500 数据集处理器"""

    def load_and_split(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        # Math-500 数据集加载 - 注意这个数据集只有 test split
        dataset = load_dataset(self.dataset_name, split="test")
        
        # 获取数据集大小
        total_size = len(dataset)
        print(f"Math-500 total size: {total_size}")
        
        # 对于500个样本的数据集，调整划分策略
        # 默认测试集300个，训练集200个
        if total_size <= 500:
            # 确保测试集不超过总量的80%，至少保留100个训练样本
            max_test_size = total_size - 100
            actual_test_size = min(test_size, max_test_size)
            
            # 如果请求的测试集太大，给出警告
            if test_size > actual_test_size:
                print(f"Warning: Requested test_size {test_size} is too large for dataset size {total_size}")
                print(f"Adjusted test_size to {actual_test_size} to ensure at least 100 training samples")
        else:
            # 对于更大的数据集（虽然Math-500应该正好是500）
            actual_test_size = min(test_size, total_size)
            if total_size < test_size + 100:
                actual_test_size = min(test_size, max(50, int(total_size * 0.6)))
                print(f"Adjusted test_size from {test_size} to {actual_test_size} due to dataset size")
        
        # 使用缓存的划分
        cache_path = self._get_split_cache_path(actual_test_size, seed)
        cached = self._load_split_indices(cache_path)
        
        if cached:
            train_indices, test_indices = cached
            print(f"Using cached split for Math-500: train={len(train_indices)}, test={len(test_indices)}")
        else:
            # 创建固定的划分
            np.random.seed(seed)
            indices = np.random.permutation(total_size).tolist()
            test_indices = indices[:actual_test_size]
            train_indices = indices[actual_test_size:]
            self._save_split_indices(train_indices, test_indices, cache_path)
            print(f"Created new split for Math-500: train={len(train_indices)}, test={len(test_indices)}")
        
        train = dataset.select(train_indices)
        test = dataset.select(test_indices)
        
        return train, test
    
    def format_example_io(self, example: Dict) -> Tuple[str, str]:
        """格式化为 I/O 模式的 (question, answer)"""
        problem = example["problem"].strip()
        answer = example["answer"].strip()
        
        # 清理答案中的LaTeX格式（保留主要内容）
        # answer = self._clean_latex_answer(answer)
        
        return problem, answer
    
    def format_example_cot(self, example: Dict) -> Tuple[str, str]:
        """格式化为 COT 模式的 (question, answer_with_reasoning)"""
        problem = example["problem"].strip()
        solution = example["solution"].strip()
        answer = example["answer"].strip()
        
        # 清理解决方案中的图形代码
        # solution = self._clean_solution(solution)
        
        # 组合解决方案和最终答案
        full_answer = f"{solution}. So the final answer is boxed{{{answer}}}."
        
        return problem, full_answer
    
    def extract_gold_answer(self, example: Dict) -> str:
        """提取标准答案"""
        answer = example["answer"].strip()
        return self._clean_latex_answer(answer).replace(" ", "")
    
    def extract_prediction(self, model_output: str) -> str:
        """从模型输出中提取预测答案"""
        import re
        
        # 优先查找 \boxed{} 格式
        boxed_content = self._extract_boxed_content(model_output)
        if boxed_content:
            return self._clean_latex_answer(boxed_content).replace(" ", "")
        
        # 查找 "answer is" 或 "answer:" 模式
        answer_patterns = [
            r"answer\s*is\s*[:\s]*(.+?)(?:\n|$)",
            r"answer\s*[:=]\s*(.+?)(?:\n|$)",
            r"final\s+answer\s*[:=]\s*(.+?)(?:\n|$)",
            r"therefore\s*,?\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, model_output, re.IGNORECASE)
            if match:
                return self._clean_latex_answer(match.group(1).strip()).replace(" ", "")
        
        # 如果找不到特定模式，尝试提取最后一个数学表达式
        math_expressions = re.findall(r'\$([^$]+)\$', model_output)
        if math_expressions:
            return self._clean_latex_answer(math_expressions[-1]).replace(" ", "")
        
        # 最后返回原始输出的最后一行非空内容
        lines = model_output.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return self._clean_latex_answer(line.strip()).replace(" ", "")
        
        return model_output.strip().replace(" ", "")
    
    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """提取 \boxed{} 中的内容，正确处理嵌套的花括号"""
        import re
        
        # 找到 \boxed{ 的位置
        start_pattern = r"\\boxed\{"
        match = re.search(start_pattern, text)
        if not match:
            return None
        
        # 从 \boxed{ 后面开始
        start_pos = match.end()
        
        # 计数花括号，找到匹配的右花括号
        brace_count = 1
        pos = start_pos
        
        while pos < len(text) and brace_count > 0:
            if text[pos] == '\\' and pos + 1 < len(text):
                # 跳过转义字符
                pos += 2
                continue
            elif text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            # 找到了匹配的右花括号
            return text[start_pos:pos - 1]
        else:
            # 没有找到匹配的右花括号
            return None
    
    def _clean_latex_answer(self, answer: str) -> str:
        """清理LaTeX格式的答案，保留主要内容"""
        if not answer:
            return ""
        
        # 移除外层的 $ 符号
        answer = answer.strip()
        if answer.startswith('$') and answer.endswith('$'):
            answer = answer[1:-1]
        
        # 移除 \boxed{} 包装
        if answer.startswith('\\boxed{') and answer.endswith('}'):
            answer = answer[7:-1]
        
        # 基本的LaTeX命令清理（保留主要数学内容）
        # 这里只做最基础的清理，保留大部分LaTeX命令以便正确比较
        answer = answer.strip()
        
        return answer
    
    def _clean_solution(self, solution: str) -> str:
        """清理解决方案，移除图形代码等"""
        import re
        
        # 移除 [asy]...[/asy] 代码块
        solution = re.sub(r'\[asy\].*?\[/asy\]', '', solution, flags=re.DOTALL)
        
        # 移除多余的空行
        lines = solution.split('\n')
        cleaned_lines = [line for line in lines if line.strip()]
        
        return '\n'.join(cleaned_lines).strip()


class UGPhysicsHandler(BaseDatasetHandler):
    """UGPhysics 数据集处理器"""

    # UGPhysics 的所有子集（按大小排序）
    ALL_SUBSETS = [
        "QuantumMechanics",
        "AtomicPhysics",
        "ClassicalMechanics",
        "StatisticalMechanics",
        "ClassicalElectromagnetism",
        "Thermodynamics",
        "TheoreticalMechanics",
        "WaveOptics",
        "Relativity",
        "SemiconductorPhysics",
        "Electrodynamics",
        "Solid-StatePhysics",
        "GeometricalOptics"
    ]

    def __init__(self, dataset_name: str, subset: str = None, cache_dir: str = "./dataset_splits"):
        super().__init__(dataset_name, subset, cache_dir)
        # UGPhysics 必须指定 subset
        if not subset:
            raise ValueError(
                f"UGPhysics requires a subset to be specified. "
                f"Use 'all' or 'mixed' to mix all subsets, or choose from: {', '.join(self.ALL_SUBSETS)}"
            )

    def _save_split_indices_with_size(self, train_indices, test_indices, test_size, cache_path):
        """保存划分索引时包含test_size信息"""
        cache_data = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'test_size': test_size,
            'train_size': len(train_indices),
            'actual_test_size': len(test_indices)  # 保存实际的测试集大小
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def _load_split_indices_with_size(self, cache_path, expected_test_size):
        """加载划分索引并验证test_size"""
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # 检查test_size是否匹配
            cached_test_size = cache_data.get('test_size')
            if cached_test_size != expected_test_size:
                print(f"Cache test_size {cached_test_size} doesn't match expected {expected_test_size}")
                return None

            print(f"Found valid cache with test_size={cached_test_size}")
            return cache_data['train_indices'], cache_data['test_indices']
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def load_and_split(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        """加载并划分数据集"""
        print(f"Requested test_size: {test_size}")

        if self.subset in ["all", "mixed"]:
            return self._load_and_split_mixed(test_size, seed)

        # 单个子集的处理
        dataset = load_dataset(self.dataset_name, self.subset, split="en")
        total_size = len(dataset)
        print(f"UGPhysics/{self.subset} total size: {total_size}")

        # 如果数据集太小，调整 test_size
        original_test_size = test_size
        if total_size < test_size + 50:  # 至少保留50个训练样本
            test_size = min(test_size, max(20, int(total_size * 0.2)))  # 测试集最多占20%
            print(f"Adjusted test_size from {original_test_size} to {test_size} due to small dataset size")

        # 使用缓存的划分（现在包含test_size在路径中）
        cache_path = self._get_split_cache_path(test_size, seed)
        print(f"Cache path: {cache_path}")

        cached = self._load_split_indices_with_size(cache_path, test_size)

        if cached:
            train_indices, test_indices = cached
            print(f"Using cached split: train={len(train_indices)}, test={len(test_indices)}")
        else:
            print(f"Creating new split with test_size={test_size}")
            # 创建固定的划分
            np.random.seed(seed)
            indices = np.random.permutation(total_size).tolist()
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            self._save_split_indices_with_size(train_indices, test_indices, test_size, cache_path)
            print(f"Created new split: train={len(train_indices)}, test={len(test_indices)}")

        train = dataset.select(train_indices)
        test = dataset.select(test_indices)

        print(f"Final split sizes: train={len(train)}, test={len(test)}")

        return train, test

    def _load_and_split_mixed(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        """加载并混合所有子集"""
        print(f"Loading and mixing all UGPhysics subsets with test_size={test_size}...")

        # 特殊的缓存路径用于混合数据集（现在包含test_size）
        cache_path = self._get_mixed_dataset_cache_path(test_size, seed)
        print(f"Mixed dataset cache path: {cache_path}")

        # 检查是否有缓存的混合数据集
        if os.path.exists(cache_path):
            print(f"Found cached mixed dataset")
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # 验证test_size是否匹配
            cached_test_size = cache_data.get('test_size', cache_data.get('actual_test_size', 0))
            if cached_test_size == test_size or len(cache_data.get('test_indices', [])) == test_size:
                print(f"Cache is valid with test_size={cached_test_size}")
                train_indices = cache_data['train_indices']
                test_indices = cache_data['test_indices']
                subset_info = cache_data['subset_info']

                # 重新加载数据集
                all_datasets = []
                for subset_name, subset_size in subset_info:
                    try:
                        ds = load_dataset(self.dataset_name, subset_name, split="en")
                        # 添加子集标签
                        ds = ds.add_column("subset", [subset_name] * len(ds))
                        all_datasets.append(ds)
                    except Exception as e:
                        print(f"Warning: Failed to load subset {subset_name}: {e}")

                # 合并所有数据集
                mixed_dataset = concatenate_datasets(all_datasets)

                # 应用缓存的索引
                train = mixed_dataset.select(train_indices)
                test = mixed_dataset.select(test_indices)

                print(f"Loaded cached split: train={len(train)}, test={len(test)}")

                return train, test
            else:
                print(f"Cache test_size {cached_test_size} doesn't match expected {test_size}, regenerating...")

        # 如果没有缓存或test_size不匹配，创建新的混合数据集
        print("Creating new mixed dataset split...")
        all_datasets = []
        subset_info = []

        for subset in self.ALL_SUBSETS:
            try:
                ds = load_dataset(self.dataset_name, subset, split="en")
                size = len(ds)
                print(f"  - {subset}: {size} examples")

                # 添加子集标签
                ds = ds.add_column("subset", [subset] * size)
                all_datasets.append(ds)
                subset_info.append((subset, size))
            except Exception as e:
                print(f"Warning: Failed to load subset {subset}: {e}")

        # 合并所有数据集
        mixed_dataset = concatenate_datasets(all_datasets)
        total_size = len(mixed_dataset)
        print(f"Total mixed dataset size: {total_size} examples")

        # 创建随机划分
        np.random.seed(seed)
        indices = np.random.permutation(total_size).tolist()

        # 确保测试集大小合理
        actual_test_size = test_size
        if total_size < test_size + 100:
            actual_test_size = min(test_size, max(100, int(total_size * 0.1)))
            print(f"Adjusted test_size from {test_size} to {actual_test_size} for mixed dataset")

        test_indices = indices[:actual_test_size]
        train_indices = indices[actual_test_size:]

        # 保存混合数据集的信息和索引
        cache_data = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'subset_info': subset_info,
            'total_size': total_size,
            'test_size': actual_test_size,  # 保存实际使用的test_size
            'train_size': len(train_indices),
            'actual_test_size': len(test_indices),  # 冗余但明确
            'seed': seed
        }

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

        train = mixed_dataset.select(train_indices)
        test = mixed_dataset.select(test_indices)

        # 打印各子集在训练集和测试集中的分布
        print("\nSubset distribution in train/test:")
        for split_name, split_data in [("Train", train), ("Test", test)]:
            subset_counts = {}
            for example in split_data:
                subset = example['subset']
                subset_counts[subset] = subset_counts.get(subset, 0) + 1

            print(f"\n{split_name} set ({len(split_data)} total):")
            for subset, count in sorted(subset_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {subset}: {count} ({count / len(split_data) * 100:.1f}%)")

        return train, test

    def _get_split_cache_path(self, test_size: int, seed: int) -> str:
        """生成包含test_size的缓存路径"""
        # 重写父类方法，确保test_size被包含在缓存路径中
        cache_id = f"{self.dataset_name}_{self.subset}_{test_size}_{seed}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"split_{cache_hash}.json")

    def _get_mixed_dataset_cache_path(self, test_size: int, seed: int) -> str:
        """生成混合数据集的缓存路径（包含test_size）"""
        cache_id = f"{self.dataset_name}_mixed_all_{test_size}_{seed}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"mixed_{cache_hash}.json")

    def format_example_io(self, example: Dict) -> Tuple[str, str]:
        """格式化输入输出示例"""
        problem = example["problem"].strip()
        answer = example["answers"].strip()
        unit = example.get("unit", "").strip()

        if unit and unit != "null" and unit != "None":
            return problem, f"{answer} {unit}"
        return problem, answer

    def format_example_cot(self, example: Dict) -> Tuple[str, str]:
        """格式化链式思维示例"""
        problem = example["problem"].strip()
        solution = example["solution"].strip()
        answer = example["answers"].strip()

        # 检查 unit 是否存在且为字符串类型
        unit = example.get("unit")
        if isinstance(unit, str):
            unit = unit.strip()
        else:
            unit = None

        if unit and unit != "null" and unit != "None":
            full_answer = f"{solution}\nFinal answer: {answer} {unit}"
        else:
            full_answer = f"{solution}\nFinal answer: {answer}"

        return problem, full_answer

    def extract_boxed_content(self, text):
        """提取 \boxed{} 中的内容，正确处理嵌套的花括号"""
        import re

        # 找到 \boxed{ 的位置
        start_pattern = r"\\boxed\{"
        match = re.search(start_pattern, text)
        if not match:
            return None

        # 从 \boxed{ 后面开始
        start_pos = match.end()

        # 计数花括号，找到匹配的右花括号
        brace_count = 1
        pos = start_pos

        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            # 找到了匹配的右花括号
            return text[start_pos:pos - 1]
        else:
            # 没有找到匹配的右花括号
            return None

    def extract_gold_answer(self, example: Dict) -> str:
        """提取标准答案"""
        answer = example["answers"].strip()
        boxed_content = self.extract_boxed_content(answer)
        if boxed_content:
            return boxed_content
        return answer

    def extract_prediction(self, model_output: str) -> str:
        """从模型输出中提取预测答案"""
        import re

        # 查找 boxed 格式
        boxed_content = self.extract_boxed_content(model_output)
        if boxed_content:
            return boxed_content

        # 查找 "Final answer:" 模式
        final_pattern = r"Final answer:\s*(.+?)(?:\n|$)"
        match = re.search(final_pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 尝试提取最后的数值
        from metrics_utils import parse_number_from_text
        num = parse_number_from_text(model_output)
        if num is not None:
            return str(num)

        return model_output.strip()

class SVAMPHandler(BaseDatasetHandler):
    """SVAMP 数据集处理器"""

    def load_and_split(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        # SVAMP 数据集有 train 和 test splits
        train_dataset = load_dataset(self.dataset_name, split="train")
        test_dataset = load_dataset(self.dataset_name, split="test")
        
        # 获取数据集大小
        train_size = len(train_dataset)
        test_size_original = len(test_dataset)
        print(f"SVAMP train size: {train_size}, test size: {test_size_original}")
        
        # 如果测试集已经足够小，直接使用
        if test_size_original <= test_size:
            print(f"Using full test set with {test_size_original} samples")
            return train_dataset, test_dataset
        
        # 否则，从测试集中采样
        # 使用缓存的划分
        cache_path = self._get_split_cache_path(test_size, seed)
        cached = self._load_split_indices(cache_path)
        
        if cached:
            _, test_indices = cached
            print(f"Using cached test subset: {len(test_indices)} samples from {test_size_original}")
            test_subset = test_dataset.select(test_indices)
        else:
            # 创建固定的测试子集
            np.random.seed(seed)
            test_indices = np.random.permutation(test_size_original).tolist()[:test_size]
            # 保存时，train_indices 传入空列表，因为我们使用完整的训练集
            self._save_split_indices([], test_indices, cache_path)
            print(f"Created test subset: {len(test_indices)} samples from {test_size_original}")
            test_subset = test_dataset.select(test_indices)
        
        return train_dataset, test_subset
    
    def format_example_io(self, example: Dict) -> Tuple[str, str]:
        """格式化为 I/O 模式的 (question, answer)"""
        question = example["question_concat"].strip()
        answer = str(example["Answer"]).strip()
        
        return question, answer
    
    def format_example_cot(self, example: Dict) -> Tuple[str, str]:
        """SVAMP 没有 COT 版本，返回与 IO 相同的格式"""
        return self.format_example_io(example)
    
    def extract_gold_answer(self, example: Dict) -> str:
        """提取标准答案"""
        return str(example["Answer"]).strip()
    
    def extract_prediction(self, model_output: str) -> str:
        """从模型输出中提取预测答案"""
        import re
        
        # 策略1: 查找 #### 格式（类似GSM8K）
        gsm_match = re.search(r'####\s*(\S+)', model_output)
        if gsm_match:
            return gsm_match.group(1).strip()
        
        # 策略2: 查找 "answer is" 或 "answer:" 模式
        answer_patterns = [
            r"answer\s*is\s*[:\s]*(\d+(?:\.\d+)?)",
            r"answer\s*[:=]\s*(\d+(?:\.\d+)?)",
            r"=\s*(\d+(?:\.\d+)?)\s*$",  # 等号后面的数字（在行末）
            r"(?:total|result|sum)\s*(?:is|=)\s*(\d+(?:\.\d+)?)",
            r"therefore\s*,?\s*(\d+(?:\.\d+)?)",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, model_output, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # 策略3: 提取最后一个独立的数字
        # 查找所有的数字
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', model_output)
        if numbers:
            # 返回最后一个数字
            return numbers[-1]
        
        # 策略4: 查找最后一行的内容
        lines = model_output.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line:
                # 尝试从最后一行提取数字
                num_match = re.search(r'\d+(?:\.\d+)?', line)
                if num_match:
                    return num_match.group()
        
        # 如果都失败了，返回原始输出
        return model_output.strip()

class ModelScopeCOTHandler(BaseDatasetHandler):
    def _hf_from_msdataset(self, ms_ds):
        data_list = []
        for item in ms_ds:
            data_list.append(dict(item))
        return Dataset.from_list(data_list)

    def _filter_by_tasks(self, ds_list, task_list):
        if not task_list:
            return ds_list
        task_set = set(task_list)
        return [ex for ex in ds_list if str(ex.get("task", "")).strip() in task_set]

    def load_and_split(self, test_size: int = 300, seed: int = 42) -> Tuple[Dataset, Dataset]:
        # 直接从本地加载数据集
        local_path = "/data/oujie/models/AI-ModelScope/CoT-Collection"
        print(f"从本地加载数据集: {local_path}")
        
        try:
            # 使用 HuggingFace datasets 从本地路径加载
            from datasets import load_dataset as hf_load_dataset
            hf_dataset = hf_load_dataset(
                local_path,
                "en",
                split="train",
                trust_remote_code=True
            )
            train_ms = hf_dataset
            print(f"成功加载数据集，总数据量: {len(train_ms)}")
        except Exception as e:
            print(f"本地加载失败: {e}")
            raise RuntimeError(f"无法从本地路径加载数据集: {local_path}")
        
        full_list = [dict(x) for x in train_ms]
        task_list = getattr(self, "task_list", None)
        print(f"task list:{task_list}")
        full_list = self._filter_by_tasks(full_list, task_list)
        total_size = len(full_list)
        actual_test_size = min(test_size, total_size)
        tasks = sorted(set([str(t).strip() for t in (task_list or []) if str(t).strip()]))
        task_sig = hashlib.md5(",".join(tasks).encode()).hexdigest()[:8] if tasks else "none"
        cache_id = f"{self.dataset_name}_{self.subset or 'default'}_{actual_test_size}_{seed}_{task_sig}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()[:8]
        cache_path = os.path.join(self.cache_dir, f"split_{cache_hash}.json")
        cached = self._load_split_indices(cache_path)
        if cached:
            train_indices, test_indices = cached
            max_idx = max([max(train_indices) if train_indices else -1, max(test_indices) if test_indices else -1])
            if max_idx >= total_size:
                train_indices, test_indices = None, None
        else:
            np.random.seed(seed)
            indices = np.random.permutation(total_size).tolist()
            test_indices = indices[:actual_test_size]
            train_indices = indices[actual_test_size:]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'total_size': total_size,
                    'task_sig': task_sig,
                    'dataset_name': self.dataset_name,
                    'subset': self.subset or 'default',
                    'seed': seed,
                    'test_size': actual_test_size
                }, f, indent=2)
        train_list = [full_list[i] for i in train_indices]
        test_list = [full_list[i] for i in test_indices]
        return Dataset.from_list(train_list), Dataset.from_list(test_list)

    def format_example_io(self, example: Dict) -> Tuple[str, str]:
        q = str(example.get("source", "")).strip()
        a = str(example.get("target", "")).strip()
        return q, a

    def format_example_cot(self, example: Dict) -> Tuple[str, str]:
        q = str(example.get("source", "")).strip()
        r = str(example.get("rationale", "")).strip()
        t = str(example.get("target", "")).strip()
        return q, f"{r}\nFinal answer: {t}"

    def extract_gold_answer(self, example: Dict) -> str:
        return str(example.get("target", "")).strip()

    def extract_prediction(self, model_output: str) -> str:
        """
        从COT-Collection数据集的模型输出中提取预测答案
        
        支持的答案格式:
        - Final answer: xxx
        - 选择题: a, b, c, d, A, B, C, D
        - 是/否题: Yes, No, a, b
        - 数字答案
        """
        import re
        
        # 🔥 步骤 0: 预处理 - 截断到第一个 "Final answer" 后的答案，避免模型继续生成新问题
        first_final_match = re.search(r'Final\s+answer\s*:\s*(.+?)(?:\n|$)', model_output, re.IGNORECASE)
        if first_final_match:
            answer_text = first_final_match.group(1).strip()
            
            # 清理答案文本：移除尾随的无关内容
            noise_patterns = [
                r'You are an AI',
                r'Problem:',
                r'Solution:',
                r'Question:',
                r'Context:',
                r'\n\n',
            ]
            for pattern in noise_patterns:
                noise_match = re.search(pattern, answer_text, re.IGNORECASE)
                if noise_match:
                    answer_text = answer_text[:noise_match.start()].strip()
            
            return self._extract_core_answer(answer_text)
        
        # 步骤 1: 尝试匹配 "answer is xxx" 格式
        m = re.search(r'answer\s+is\s+(.+?)(?:[\n\r.]|$)', model_output, re.IGNORECASE)
        if m:
            return self._extract_core_answer(m.group(1).strip())
        
        # 步骤 2: 尝试匹配独立的选择项答案
        lines = model_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            option_match = re.match(r'^[`\(]?\s*([abcdABCD])\s*[\)`]?[\s\.\)]?(?:for\s+)?(Yes|No)?[\s\.]*$', line, re.IGNORECASE)
            if option_match:
                return option_match.group(1).lower()
            
            choice_match = re.match(r'^([abcdABCD])\s*[\)\.]?\s*(for\s+)?(Yes|No)?\.?\s*$', line, re.IGNORECASE)
            if choice_match:
                return choice_match.group(1).lower()
        
        # 步骤 3: 如果以上都失败，返回第一行非空内容
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Problem:') and not line.startswith('You are'):
                return self._extract_core_answer(line)
        
        return model_output.strip()[:100]
    
    def _extract_core_answer(self, answer_text: str) -> str:
        """从答案文本中提取核心答案"""
        import re
        
        answer_text = answer_text.strip()
        answer_text = re.sub(r'[\.,;:!?]+$', '', answer_text).strip()
        
        # 匹配选择项格式
        option_match = re.match(r'^[`\(]?\s*([abcdABCD1234])\s*[\)`]?(?:\s*[\)\.]\s*(?:for\s+)?(?:Yes|No|yes|no)?)?\s*[`]?$', answer_text, re.IGNORECASE)
        if option_match:
            opt = option_match.group(1)
            if opt in '1234':
                return chr(ord('a') + int(opt) - 1)
            return opt.lower()
        
        # 匹配 "a. Western Bulldogs" 格式
        option_with_text = re.match(r'^([abcdABCD])\s*[\)\.:]\s*', answer_text)
        if option_with_text:
            return option_with_text.group(1).lower()
        
        # 匹配数字答案
        num_match = re.match(r'^(\d+(?:[,\s]\d{3})*(?:\.\d+)?)\s*$', answer_text)
        if num_match:
            return num_match.group(1).replace(',', '').replace(' ', '')
        
        # 匹配 Yes/No
        if answer_text.lower() in ['yes', 'no']:
            return answer_text.lower()
        
        # 截断过长的答案
        if len(answer_text) > 200:
            first_line = answer_text.split('\n')[0].strip()
            first_sentence = re.split(r'[.!?]', answer_text)[0].strip()
            return first_line if len(first_line) <= 100 else first_sentence[:100]
        
        return answer_text

DATASET_HANDLERS = {
    "openai/gsm8k": GSM8KHandler,
    "deepmind/aqua_rat": AquaRatHandler,
    "UGPhysics/ugphysics": UGPhysicsHandler,
    "HuggingFaceH4/MATH-500": Math500Handler,  # Math-500 数据集
    "ChilleD/SVAMP": SVAMPHandler,  # SVAMP 数据集
    "AI-ModelScope/CoT-Collection": ModelScopeCOTHandler,
}


def get_dataset_handler(dataset_name: str, subset: str = None, tasks: str = None) -> BaseDatasetHandler:
    """获取对应的数据集处理器
    
    Args:
        dataset_name: 数据集名称
        subset: 数据集子集
        tasks: 任务列表（逗号分隔），用于过滤 ModelScope CoT-Collection 等数据集
    """
    if dataset_name not in DATASET_HANDLERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    handler_class = DATASET_HANDLERS[dataset_name]
    handler = handler_class(dataset_name, subset)
    
    # 如果提供了 tasks 参数，且 handler 支持 task_list，则设置它
    if tasks and hasattr(handler, '_filter_by_tasks'):
        task_list = [t.strip() for t in tasks.split(',') if t.strip()]
        handler.task_list = task_list
    
    return handler


def list_available_datasets_and_subsets():
    """列出所有支持的数据集和它们的子集"""
    print("Available datasets and their subsets:")
    for dataset_name in DATASET_HANDLERS:
        print(f"\n{dataset_name}:")
        try:
            subsets = BaseDatasetHandler.get_available_subsets(dataset_name)
            if subsets:
                for subset in subsets:
                    print(f"  - {subset}")
            else:
                print("  - (no subsets found)")
        except Exception as e:
            print(f"  - Error: {e}")


# 如果直接运行此文件，显示可用的数据集和子集
if __name__ == "__main__":
    list_available_datasets_and_subsets()