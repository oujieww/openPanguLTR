# -*- coding: utf-8 -*-
"""
评测指标与工具：字符串标准化、Token-F1、数值解析、数值一致性与相对误差、置信区间等。
"""

import re
import string
import math
import numpy as np
from typing import Optional, Tuple

import math
import re
from typing import Optional, Union
import logging


def normalize_string(s: str) -> str:
    """标准化字符串：去除前后空格，压缩中间空格，转小写，去除标点符号"""
    # 去除前后空格
    s = s.strip()
    # 将多个连续空格替换为单个空格
    s = re.sub(r'\s+', ' ', s)
    # 转为小写
    s = s.lower()
    # 可选：去除标点符号（根据需要调整）
    # s = re.sub(r'[^\w\s]', '', s)
    return s


def flexible_equal(pred: Union[str, float], gold: Union[str, float],
                   strict: bool = False) -> bool:
    """灵活的相等比较，支持数值和字符串"""
    # 如果两者都是数值类型
    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
        return numeric_equal(float(pred), float(gold))

    # 尝试将字符串转换为数值进行比较
    try:
        pred_num = float(str(pred).strip())
        gold_num = float(str(gold).strip())
        return numeric_equal(pred_num, gold_num)
    except (ValueError, AttributeError):
        # 如果无法转换为数值，则作为字符串比较
        pred_str = str(pred)
        gold_str = str(gold)

        if strict:
            # 严格比较模式：完全相等
            return pred_str == gold_str
        else:
            # 宽松比较模式：标准化后比较
            return normalize_string(pred_str) == normalize_string(gold_str)
# ---------- 字符串规范化 ----------

def normalize_answer(s: str) -> str:
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    s = s.lower()
    s = remove_punc(s)
    s = " ".join(s.split())
    return s

def token_f1_pair(pred: str, gold: str) -> float:
    """
    简单 Token-F1（空格分词 + 去标点小写），用于对比而非主指标。
    """
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pset = {}
    for t in p:
        pset[t] = pset.get(t, 0) + 1
    gset = {}
    for t in g:
        gset[t] = gset.get(t, 0) + 1
    inter = 0
    for t, c in pset.items():
        inter += min(c, gset.get(t, 0))
    precision = inter / max(1, len(p))
    recall = inter / max(1, len(g))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ---------- GSM8K 答案解析 ----------

def extract_gold_answer_from_ref(ref_answer: str) -> str:
    """
    从 GSM8K 标注中取最终答案：按 '####' 切分并取尾段；缺省时回退为全串strip
    """
    if "####" in ref_answer:
        return ref_answer.split("####")[-1].strip()
    return ref_answer.strip()

def extract_prediction_answer(model_output: str) -> str:
    """
    从模型输出中抽取最终答案：优先取 '####' 后段；否则回退为原串 strip
    """
    if "####" in model_output:
        return model_output.split("####")[-1].strip()
    return model_output.strip()

# ---------- 数值解析与误差 ----------

_num_pattern = re.compile(
    r"""
    (?P<sign>[-+])?
    (?:
        (?P<int>\d{1,3}(?:,\d{3})+|\d+)      # 带千分位或纯整数
        (?:\.(?P<frac>\d+))?                 # 小数
        |
        (?P<num>\d+)\s*/\s*(?P<den>\d+)      # 分数 a/b
    )
    (?:\s*%){0,1}                             # 可选百分号
    """,
    re.VERBOSE
)

def parse_number_from_text(text: str) -> Optional[float]:
    """
    从文本中提取“最后出现”的数值（支持整数/小数/分数/百分数）。
    若有 % 则除以100。
    """
    clean = text.replace(",", "")
    matches = list(_num_pattern.finditer(clean))
    if not matches:
        return None
    m = matches[-1]
    if m.group("num") and m.group("den"):
        # 分数
        a = float(m.group("num"))
        b = float(m.group("den"))
        val = a / b if b != 0 else float("inf")
    else:
        integ = m.group("int")
        frac = m.group("frac")
        if integ is None:  # safety
            return None
        val = float(f"{integ}.{frac}" if frac else integ)
    sign = -1.0 if m.group("sign") == "-" else 1.0
    val *= sign
    # 百分数
    has_percent = ("%" in m.group(0))
    if has_percent:
        val /= 100.0
    return val

def numeric_equal(a: float, b: float, atol: float = 0.0, rtol: float = 0.0) -> bool:
    # 严格数值相等（允许极微小浮点误差）
    return math.isclose(a, b, abs_tol=max(atol, 1e-12), rel_tol=max(rtol, 0.0))

def relative_error(pred: float, gold: float) -> Optional[float]:
    if gold == 0:
        return abs(pred - gold)
    return abs(pred - gold) / (abs(gold) + 1e-12)

# ---------- 统计工具 ----------

def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    返回 (low, high)。n_total==0 时返回 (0,0)。
    """
    if n_total <= 0:
        return 0.0, 0.0
    from math import sqrt
    z = 1.959963984540054  # 95%
    phat = n_success / n_total
    denom = 1 + z*z/n_total
    centre = phat + z*z/(2*n_total)
    adj = z * sqrt((phat*(1-phat) + z*z/(4*n_total))/n_total)
    low = (centre - adj)/denom
    high = (centre + adj)/denom
    return max(0.0, low), min(1.0, high)

def count_tokens(tokenizer, text: str) -> int:
    enc = tokenizer(text)
    # enc.input_ids is a List[List[int]] (batch size 1); fall back to returned dict for safety
    ids = getattr(enc, 'input_ids', None)
    if ids is None:
        ids = enc.get('input_ids', [])
    return len(ids[0]) if ids else 0

def summarize_metrics():  # 预留占位（如需扩展额外汇总逻辑）
    pass
