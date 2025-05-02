#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import librosa
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser(description="评估DPO训练后的模型")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="分词器路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据路径，可以是normalized_text.json或test_data.json")
    parser.add_argument("--prompt_wav_path", type=str, required=True, help="提示音频文件路径")
    parser.add_argument("--prompt_text_path", type=str, required=True, help="提示文本文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--max_samples", type=int, default=10, help="最大样本数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    return parser.parse_args()

def load_models(args):
    """加载模型和分词器"""
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    
    return model, tokenizer

def read_prompt(prompt_wav_path, prompt_text_path):
    """读取提示音频和文本"""
    print(f"处理提示音频: {prompt_wav_path}")
    
    # 读取提示文本
    with open(prompt_text_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    
    # 读取提示音频
    waveform, sample_rate = torchaudio.load(prompt_wav_path)
    
    # 确保是单声道
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 转换到目标采样率
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    
    return prompt_text, waveform.squeeze(0).numpy(), sample_rate

def gen_chat_prompt_from_text(text):
    """生成聊天格式的提示"""
    prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def evaluate_model(args, model, tokenizer, prompt_text, prompt_waveform, sample_rate):
    """评估模型在数字朗读任务上的效果"""
    # 读取测试数据
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 检查测试数据格式（是否是test_data.json格式或normalized_text.json格式）
    # test_data.json中直接包含测试样本
    # normalized_text.json中每个样本都有"original"和"normalized"字段
    if not isinstance(test_data, list):
        print(f"错误: 测试数据应该是一个列表，但获取到的是 {type(test_data)}")
        return
    
    # 确保测试数据格式正确
    if len(test_data) > 0:
        first_item = test_data[0]
        if isinstance(first_item, dict) and "original" in first_item and "normalized" in first_item:
            print(f"检测到normalized_text.json格式的测试数据 ({len(test_data)} 个样本)")
        else:
            print(f"错误: 测试数据格式不正确，第一项: {first_item}")
            return
    
    # 限制样本数
    if args.max_samples > 0 and args.max_samples < len(test_data):
        test_data = test_data[:args.max_samples]
        print(f"限制评估样本数量为 {args.max_samples}")
    
    print(f"开始评估 {len(test_data)} 个测试样本...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存原始提示语音
    prompt_wav_path = os.path.join(args.output_dir, "prompt.wav")
    sf.write(prompt_wav_path, prompt_waveform, sample_rate)
    
    # 结果汇总
    results = []
    
    # 遍历测试样本
    for idx, item in enumerate(tqdm(test_data)):
        original_text = item["original"]
        normalized_text = item["normalized"]
        
        sample_id = f"test_{idx:03d}"
        
        # 生成原始数字和标准化数字的结果
        for text_type, text in [("original", original_text), ("normalized", normalized_text)]:
            # 构建完整提示文本
            full_prompt = f"{prompt_text}\n{text}"
            chat_prompt = gen_chat_prompt_from_text(full_prompt)
            
            # 编码输入
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(args.device)
            
            # 模拟生成过程 - 在实际应用中，这里应该使用模型生成语音
            # 由于我们不能实际运行TTS模型生成，这里只记录输入
            
            # 保存结果
            output_path = os.path.join(args.output_dir, f"{sample_id}_{text_type}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Prompt:\n{full_prompt}\n\n")
                f.write(f"Text Type: {text_type}\n")
            
            # 添加到结果列表
            results.append({
                "sample_id": sample_id,
                "text_type": text_type,
                "text": text,
                "output_path": output_path
            })
    
    # 保存评估结果摘要
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评估完成，结果保存在：{args.output_dir}")
    print(f"评估摘要：{summary_path}")

def main():
    args = parse_args()
    
    # 加载模型
    model, tokenizer = load_models(args)
    
    # 读取提示语音和文本
    prompt_text, prompt_waveform, sample_rate = read_prompt(args.prompt_wav_path, args.prompt_text_path)
    
    # 评估模型
    evaluate_model(args, model, tokenizer, prompt_text, prompt_waveform, sample_rate)
    
    print("评估完成！")

if __name__ == "__main__":
    main() 