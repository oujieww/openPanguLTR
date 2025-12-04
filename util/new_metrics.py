import re
import string
import math
from typing import Optional, Tuple


def normalize_answer(s: str) -> str:
    """标准化答案：转小写、去除标点、压缩空格等"""
    if not isinstance(s, str):
        s = str(s)

    # 转换为小写
    s = s.lower()

    # 移除标点符号（但保留数字中的小数点、负号和逗号）
    s = re.sub(r'[^\w\s.,-]', ' ', s)

    # 移除多余的空格
    s = ' '.join(s.split())

    # 去除首尾空格
    s = s.strip()

    return s


def parse_number_from_text(text: str) -> Optional[float]:
    """从文本中提取数字，支持多种格式"""
    if not isinstance(text, str):
        text = str(text)

    # 移除常见的数值修饰词
    modifiers = [
        'approximately', 'about', 'around', 'nearly', 'roughly',
        'exactly', 'precisely', 'approx', '~', '≈',
        'the answer is', 'answer:', 'result:', 'equals', '=',
        'is equal to', 'totals', 'amounts to', 'sums to'
    ]

    cleaned_text = text.lower()
    for modifier in modifiers:
        cleaned_text = cleaned_text.replace(modifier, ' ')

    # 先尝试直接转换（处理纯数字的情况）
    try:
        # 移除逗号后尝试转换
        cleaned = cleaned_text.strip().replace(',', '')
        return float(cleaned)
    except:
        pass

    # 使用正则表达式查找数字
    patterns = [
        r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # 带逗号的数字
        r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',  # 科学计数法
        r'[-+]?\d+/\d+',  # 分数
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_text)
        all_matches.extend(matches)

    if all_matches:
        # 选择最长的匹配（通常是最完整的数字）
        all_matches.sort(key=len, reverse=True)
        for match in all_matches:
            try:
                # 移除逗号
                num_str = match.replace(',', '')

                # 跳过太短的匹配
                if len(num_str.replace('.', '').replace('-', '')) < 1:
                    continue

                # 处理分数
                if '/' in num_str:
                    parts = num_str.split('/')
                    return float(parts[0]) / float(parts[1])

                return float(num_str)
            except:
                continue

    return None


def numeric_equal(a: float, b: float, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    """数值相等判断（允许合理的浮点误差）"""
    return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def relative_error(pred: float, gold: float) -> Optional[float]:
    """计算相对误差"""
    if gold == 0:
        return abs(pred - gold) if pred != 0 else 0.0
    return abs(pred - gold) / abs(gold)


def token_f1_pair(pred: str, gold: str) -> float:
    """计算token级别的F1分数"""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

def normalize_latex_expression(expr: str) -> str:
    """标准化 LaTeX 数学表达式以便比较"""
    import re
    
    if not expr:
        return ""
    
    # 移除外层的 $ 符号
    expr = expr.strip()
    if expr.startswith('$') and expr.endswith('$'):
        expr = expr[1:-1]
    
    # 标准化空格
    expr = re.sub(r'\s+', ' ', expr)
    
    # 标准化 \left 和 \right
    expr = expr.replace('\\left(', '(').replace('\\right)', ')')
    expr = expr.replace('\\left[', '[').replace('\\right]', ']')
    expr = expr.replace('\\left\\{', '\\{').replace('\\right\\}', '\\}')
    
    # 标准化分数表示
    expr = expr.replace('\\tfrac', '\\frac')
    expr = expr.replace('\\dfrac', '\\frac')
    
    # 移除不影响数学意义的命令
    expr = expr.replace('\\displaystyle', '')
    expr = expr.replace('\\,', ' ')
    expr = expr.replace('\\:', ' ')
    expr = expr.replace('\\;', ' ')
    expr = expr.replace('\\!', '')
    
    # 再次清理多余空格
    expr = ' '.join(expr.split())
    
    return expr.strip()

def evaluate_answer(pred_final: str, gold_final: str) -> dict:
    """综合评估答案，支持文本和数值"""
    results = {}

    # 首先尝试解析数值
    gold_num = parse_number_from_text(gold_final)
    pred_num = parse_number_from_text(pred_final)

    # 初始化数值相关结果
    results['numeric_ok'] = 0
    results['relerr'] = None
    results['has_numeric'] = (gold_num is not None)

    # 数值比较
    if gold_num is not None and pred_num is not None:
        results['numeric_ok'] = 1 if numeric_equal(pred_num, gold_num) else 0
        results['relerr'] = relative_error(pred_num, gold_num)

    # 文本级别的评估
    # 1. 精确匹配（标准化后）
    results['em_str'] = 1.0 if normalize_answer(pred_final) == normalize_answer(gold_final) else 0.0

    # 2. 包含匹配（用于答案可能包含额外信息的情况）
    results['contains'] = 1.0 if normalize_answer(gold_final) in normalize_answer(pred_final) else 0.0

    # 3. Token F1 分数
    results['tf1'] = token_f1_pair(pred_final, gold_final)

    # 综合评分策略
    if results['has_numeric'] and results['numeric_ok'] == 1:
        # 数值任务且数值正确：完全正确
        results['em_str'] = 1.0
        results['overall_score'] = 1.0
    elif results['has_numeric'] and pred_num is not None:
        # 数值任务但数值不正确
        results['overall_score'] = results['numeric_ok']
    elif results['has_numeric']:
        # 数值任务但未能提取预测中的数值
        results['overall_score'] = 0.0
    else:
        # 纯文本任务：综合考虑多种匹配方式
        results['overall_score'] = max(
            results['em_str'],
            results['contains'],  # 包含匹配稍微降低权重
            results['tf1']  # F1匹配进一步降低权重
        )

    return results


# 测试
if __name__ == "__main__":
    print("测试数值处理:")
    numeric_tests = [
        ("1,234.56", "1234.56"),
        ("approximately 1,234.56", "1234.56"),
        ("The answer is 42", "42"),
        ("Result: -123.45", "-123.45"),
        ("About $1,000,000", "1000000"),
        ("1234.56", "1234.561"),
        ("1233.56", "1234.561"),
        ("1233.56", "1234.551"),
    ]

    for pred, gold in numeric_tests:
        result = evaluate_answer(pred, gold)
        print(f"{pred} vs {gold}:")
        print(f"  numeric_ok: {result['numeric_ok']}, em_str: {result['em_str']}, overall: {result['overall_score']}")

    print("\n测试文本处理:")
    text_tests = [
        ("Beijing", "beijing"),
        ("The capital is Beijing.", "Beijing"),
        ("PARIS", "paris"),
        ("New York City!", "new york city"),
        ("machine learning", "Machine Learning"),
        ("Price: $99.99", "99.99"),
        ("The population is approximately 1,234,567", "1234567"),
    ]

    for pred, gold in text_tests:
        result = evaluate_answer(pred, gold)
        print(f"{pred} vs {gold}:")
        print(
            f"  em_str: {result['em_str']}, contains: {result['contains']}, tf1: {result['tf1']:.2f}, overall: {result['overall_score']:.2f}")
