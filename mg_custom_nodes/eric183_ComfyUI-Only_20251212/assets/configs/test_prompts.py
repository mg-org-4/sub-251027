"""
测试提示词分类逻辑
"""

def is_negative_prompt(text):
    """
    判断文本是否为negative prompt
    """
    text_lower = text.lower()
    
    # 首先检查明确的positive关键词
    positive_keywords = ["masterpiece", "best quality", "best"]
    for keyword in positive_keywords:
        if keyword in text_lower:
            return False  # 明确是positive
    
    # 然后检查明确的negative关键词
    negative_keywords = ["worst", "bad"]
    for keyword in negative_keywords:
        if keyword in text_lower:
            return True   # 明确是negative
    
    # 如果没有明确关键词，使用其他特征判断
    # 以lora标签开头通常是positive
    if text.strip().startswith("<lora:"):
        return False
    
    # 包含更多negative特征词汇
    extended_negative_keywords = [
        "low quality", "normal quality", "bad anatomy", "bad hands", 
        "watermark", "signature", "simple background", "transparent"
    ]
    for keyword in extended_negative_keywords:
        if keyword in text_lower:
            return True
    
    # 默认判断为positive（保守策略）
    return False

# 测试用例
test_cases = [
    # Positive 提示词
    ("<lora:AddMicroDetails_Illustrious_v4:0.4>,addmicrodetails,\nmasterpiece, best quality, amazing quality", False),
    ("masterpiece, best quality, high resolution", False),
    ("<lora:some_model:1.0>, beautiful girl", False),
    ("best art, detailed background", False),
    
    # Negative 提示词
    ("worst quality,normal quality,anatomical nonsense,bad anatomy", True),
    ("bad hands, bad fingers, worst quality", True),
    ("low quality, watermark, signature", True),
    ("simple background, transparent", True),
]

if __name__ == "__main__":
    print("🧪 测试提示词分类逻辑")
    print("=" * 50)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        result = is_negative_prompt(text)
        status = "✅" if result == expected else "❌"
        prompt_type = "Negative" if result else "Positive"
        expected_type = "Negative" if expected else "Positive"
        
        print(f"{status} 测试 {i}: {prompt_type}")
        print(f"   预期: {expected_type}")
        print(f"   文本: {text[:50]}...")
        print()
    
    print("测试完成！") 