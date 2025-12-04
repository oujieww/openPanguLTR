"""
模型兼容性工具
处理不同模型的特殊需求
"""


def get_eos_token_ids(tokenizer):
    """
    获取 EOS token IDs，兼容不同模型
    
    Args:
        tokenizer: HuggingFace tokenizer
    
    Returns:
        eos_ids: EOS token ID 或 ID 列表
    """
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    
    # Llama 3.1 等模型可能有多个 EOS token
    if hasattr(tokenizer, "eos_token_id") and isinstance(tokenizer.eos_token_id, list):
        eos_ids = tokenizer.eos_token_id
    
    return eos_ids


def need_trust_remote_code(model_name: str) -> bool:
    """
    判断模型是否需要 trust_remote_code 参数
    
    Args:
        model_name: 模型名称或路径
    
    Returns:
        bool: 是否需要 trust_remote_code
    """
    model_name_lower = model_name.lower()
    
    # 需要 trust_remote_code 的模型
    need_trust_models = [
        "qwen",      # Qwen 系列
        "chatglm",   # ChatGLM 系列
        "baichuan",  # Baichuan 系列
    ]
    
    return any(model in model_name_lower for model in need_trust_models)


def build_chat_prompt(tokenizer, messages):
    """
    构建对话 prompt，兼容不同模型的 chat template
    
    Args:
        tokenizer: HuggingFace tokenizer
        messages: 对话消息列表
    
    Returns:
        prompt_text: 构建的 prompt 文本
    """
    try:
        # 尝试使用 tokenizer 的 chat_template
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt_text
    except Exception:
        # 降级到简单拼接
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        if system_content:
            return f"{system_content}\n\n{user_content}\n"
        else:
            return f"{user_content}\n"


def get_model_info(model_name: str) -> dict:
    """
    获取模型信息
    
    Args:
        model_name: 模型名称或路径
    
    Returns:
        dict: 模型信息
    """
    model_name_lower = model_name.lower()
    
    info = {
        "name": model_name,
        "need_trust_remote": need_trust_remote_code(model_name),
        "type": "unknown"
    }
    
    # 识别模型类型
    if "llama" in model_name_lower:
        info["type"] = "llama"
        info["multi_eos"] = True  # Llama 3.1+ 有多个 EOS token
    elif "qwen" in model_name_lower:
        info["type"] = "qwen"
        info["multi_eos"] = False
    elif "chatglm" in model_name_lower:
        info["type"] = "chatglm"
        info["multi_eos"] = False
    elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
        info["type"] = "mistral"
        info["multi_eos"] = False
    
    return info
