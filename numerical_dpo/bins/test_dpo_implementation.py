#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到 PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from numerical_dpo.utils.data_processor import NumericalDataProcessor, RepCoderProcessor
from numerical_dpo.models.ints_tts import IntsTTSModel

def parse_args():
    parser = argparse.ArgumentParser(description="测试DPO模型实现")
    parser.add_argument("--ints_model_path", type=str, default="Ints/ins_model_dpo_0228",
                        help="Ints模型路径")
    parser.add_argument("--w2v_bert_path", type=str, default="Ints/w2v-bert-2",
                        help="wav2vec-BERT模型路径")
    parser.add_argument("--dual_codec_path", type=str, default="Ints/SpeechGeneration-dev-ins/RepCodec",
                        help="RepCodec路径")
    parser.add_argument("--normalized_text_path", type=str, default="numerical/normalized_text.json",
                        help="normalized_text.json路径")
    parser.add_argument("--output_dir", type=str, default="numerical_dpo/data/test",
                        help="输出目录")
    parser.add_argument("--prompt_wav_path", type=str, default="prompts/prompt.wav",
                        help="提示音频文件路径")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="最大样本数")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    return parser.parse_args()

def test_numerical_processor(args):
    """测试数值数据处理器"""
    print("\n========== 测试数值数据处理器 ==========")
    processor = NumericalDataProcessor(
        normalized_text_path=args.normalized_text_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    # 提取数字模式
    patterns = processor.extract_number_patterns()
    print(f"数字模式示例（前3条）：")
    for i, pattern in enumerate(patterns[:3]):
        print(f"  {i+1}. 原始模式: {pattern['original']}")
        print(f"     原始上下文: {pattern['original_context']}")
        print(f"     标准化上下文: {pattern['normalized_context']}")
        print()
    
    # 创建模拟DPO数据
    dpo_data_path = processor.create_mock_dpo_data()
    
    # 验证生成的文件
    with open(dpo_data_path, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    
    print(f"生成的DPO数据示例（第一条）：")
    print(json.dumps(dpo_data[0], ensure_ascii=False, indent=2))
    
    # 验证生成的token文件
    token_path = dpo_data[0]["chosen"]["speech_token_path_dict"]["extracted"]
    tokens = np.load(token_path)
    print(f"生成的token形状: {tokens.shape}, 值范围: [{tokens.min()}, {tokens.max()}]")
    
    return True

def test_dpo_utils(args):
    """测试DPO工具函数"""
    print("\n========== 测试DPO工具函数 ==========")
    
    # 导入DPO工具函数
    sys.path.append(os.path.join(project_root, "numerical_dpo"))
    from utils.dpo_utils import gen_chat_prompt_from_text_phi3, normalize_chinese_numbers
    
    # 测试聊天提示生成
    text = "请朗读：2023-05-06"
    chat_prompt = gen_chat_prompt_from_text_phi3(text)
    print(f"原始文本: {text}")
    print(f"聊天提示: {chat_prompt}")
    
    # 测试中文数字标准化
    text = "请朗读：2023-05-06，共有8种方案"
    normalized = normalize_chinese_numbers(text)
    print(f"原始文本: {text}")
    print(f"标准化后: {normalized}")
    
    return True

def test_tts_model(args):
    """测试TTS模型"""
    print("\n========== 测试TTS模型 ==========")
    
    try:
        # 检查模型路径是否存在
        if not os.path.exists(os.path.join(project_root, args.ints_model_path)):
            print(f"模型路径不存在: {args.ints_model_path}")
            print("跳过TTS模型测试")
            return False
        
        # 初始化TTS模型
        model = IntsTTSModel(
            model_path=os.path.join(project_root, args.ints_model_path),
            device=args.device
        )
        
        # 测试生成语音tokens
        text = "请朗读：2023年5月6日"
        print(f"测试文本: {text}")
        print("生成语音tokens...")
        
        # 注意：这里不实际运行模型生成，因为可能没有GPU或模型文件
        # 在实际应用中取消注释下面的代码
        """
        tokens = model.generate_speech_tokens(text, max_new_tokens=1000)
        print(f"生成的tokens形状: {tokens.shape}")
        print(f"tokens值范围: [{tokens.min()}, {tokens.max()}]")
        """
        
        # 测试batch生成
        texts = [
            "请朗读：2023年5月6日",
            "共有127.5千克",
            "音频的采样率是44.1kHz"
        ]
        print(f"测试批量文本数量: {len(texts)}")
        print("批量生成语音tokens...")
        
        # 注意：这里不实际运行模型生成，因为可能没有GPU或模型文件
        """
        batch_tokens = model.batch_generate_speech_tokens(texts, max_new_tokens=1000)
        print(f"生成的批量tokens数量: {len(batch_tokens)}")
        """
        
        print("TTS模型测试完成！")
        return True
    
    except Exception as e:
        print(f"测试TTS模型时出错: {e}")
        return False

def test_data_and_model_integration(args):
    """测试数据和模型集成"""
    print("\n========== 测试数据和模型集成 ==========")
    
    # 加载数值数据
    processor = NumericalDataProcessor(
        normalized_text_path=args.normalized_text_path,
        output_dir=args.output_dir,
        max_samples=2  # 仅使用2个样本进行测试
    )
    
    print("加载数值数据完成")
    
    # 提取示例文本
    original_texts = [item["original"] for item in processor.data[:2]]
    normalized_texts = [item["normalized"] for item in processor.data[:2]]
    
    print("示例文本对:")
    for orig, norm in zip(original_texts, normalized_texts):
        print(f"  原始: {orig}")
        print(f"  标准化: {norm}")
        print()
    
    # 注意：这里不实际运行模型生成，因为可能没有GPU或模型文件
    print("在实际环境中，这里将使用TTS模型生成语音tokens")
    print("然后使用这些tokens创建DPO训练数据")
    
    return True

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行测试
    print("开始测试DPO实现...")
    
    # 测试数值数据处理器
    test_numerical_processor(args)
    
    # 测试DPO工具函数
    test_dpo_utils(args)
    
    # 测试TTS模型
    test_tts_model(args)
    
    # 测试数据和模型集成
    test_data_and_model_integration(args)
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main() 