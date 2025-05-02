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
from transformers import AutoTokenizer
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import random

# 添加项目根目录到 PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.dpo_utils import extract_speech_tokens_from_audio

def parse_args():
    parser = argparse.ArgumentParser(description="准备DPO训练数据")
    parser.add_argument("--ints_model_path", type=str, required=True, help="Ints模型路径")
    parser.add_argument("--w2v_bert_path", type=str, required=True, help="wav2vec-BERT模型路径")
    parser.add_argument("--dual_codec_path", type=str, required=True, help="RepCodec路径")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="分词器路径")
    parser.add_argument("--normalized_text_path", type=str, required=True, help="normalized_text.json路径")
    parser.add_argument("--prompt_wav_path", type=str, required=True, help="提示音频文件路径")
    parser.add_argument("--prompt_text_path", type=str, required=True, help="提示文本文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--output_json", type=str, required=True, help="输出JSON文件路径")
    parser.add_argument("--max_samples", type=int, default=500, help="最大样本数")
    parser.add_argument("--audio_token_shift", type=int, default=32066, help="音频token偏移值")
    parser.add_argument("--bos_audio_token_id", type=int, default=32064, help="音频开始token ID")
    parser.add_argument("--eos_audio_token_id", type=int, default=32065, help="音频结束token ID")
    parser.add_argument("--num_workers", type=int, default=4, help="并行处理的工作线程数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--mock_data", action="store_true", help="是否创建模拟数据(不使用实际模型)")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def load_models(args):
    """加载所需的模型和工具"""
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 添加RepCodec路径到系统路径
    sys.path.append(args.dual_codec_path)
    
    # 加载RepCodec
    if not args.mock_data:
        try:
            from RepCodec.RepCoder import RepCoder
            print("加载RepCodec模型...")
            repcoder = RepCoder(
                path=args.w2v_bert_path,
                device=args.device
            )
        except Exception as e:
            print(f"加载RepCodec失败: {e}")
            print("将使用模拟数据代替")
            repcoder = None
            args.mock_data = True
    else:
        repcoder = None
    
    return tokenizer, repcoder

def read_prompt(prompt_wav_path, prompt_text_path, repcoder, args):
    """读取提示音频并提取tokens"""
    print(f"处理提示音频: {prompt_wav_path}")
    
    # 读取提示文本
    with open(prompt_text_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    
    # 如果使用模拟数据，则生成随机tokens
    if args.mock_data or repcoder is None:
        print("生成模拟prompt语音tokens")
        prompt_speech_token = np.random.randint(1000, 5000, size=(1, 200)) + args.audio_token_shift
        prompt_duration = 4.0  # 假设4秒
        return prompt_text, prompt_speech_token, prompt_duration
    
    # 否则，从实际音频提取tokens
    try:
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

        # 提取语音token
        prompt_speech_token = extract_speech_tokens_from_audio(prompt_wav_path, repcoder)
        
        # 计算语音时长(秒)
        prompt_duration = waveform.size(1) / sample_rate
        
        return prompt_text, prompt_speech_token, prompt_duration
    
    except Exception as e:
        print(f"处理提示音频失败: {e}")
        print("使用模拟数据代替")
        prompt_speech_token = np.random.randint(1000, 5000, size=(1, 200)) + args.audio_token_shift
        prompt_duration = 4.0  # 假设4秒
        return prompt_text, prompt_speech_token, prompt_duration

def generate_random_tokens(length, audio_token_shift=32066, save_path=None):
    """生成随机语音tokens（用于测试）"""
    # 生成随机值
    random_tokens = np.random.randint(1000, 5000, size=(1, length))
    
    # 应用audio_token_shift
    shifted_tokens = random_tokens + audio_token_shift
    
    # 保存到文件
    if save_path is not None:
        np.save(save_path, shifted_tokens)
    
    return shifted_tokens

def process_sample(item_args):
    """处理单个样本(用于并行处理)"""
    idx, item, output_dir, args = item_args
    original_text = item["original"]
    normalized_text = item["normalized"]
    
    # 为每个样本创建唯一ID
    sample_id = f"numerical_{idx:06d}"
    
    # 创建子目录
    chosen_dir = os.path.join(output_dir, f"{sample_id}_chosen")
    rejected_dir = os.path.join(output_dir, f"{sample_id}_rejected")
    os.makedirs(chosen_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    
    # token文件路径
    chosen_token_path = os.path.join(chosen_dir, f"{sample_id}_chosen.npy")
    rejected_token_path = os.path.join(rejected_dir, f"{sample_id}_rejected.npy")
    
    # 音频文件路径
    chosen_wav_path = os.path.join(chosen_dir, f"{sample_id}_chosen.wav")
    rejected_wav_path = os.path.join(rejected_dir, f"{sample_id}_rejected.wav")
    
    # 如果已存在token文件，则跳过（实现断点续传）
    if os.path.exists(chosen_token_path) and os.path.exists(rejected_token_path):
        # 加载已存在的token文件
        chosen_speech_token = np.load(chosen_token_path)
        rejected_speech_token = np.load(rejected_token_path)
    else:
        # 生成随机tokens(用于模拟或测试)
        token_length_chosen = len(normalized_text) * 6  # 每个字符约6个token
        token_length_rejected = len(original_text) * 6
        
        chosen_speech_token = generate_random_tokens(
            token_length_chosen, 
            audio_token_shift=args.audio_token_shift,
            save_path=chosen_token_path
        )
        
        rejected_speech_token = generate_random_tokens(
            token_length_rejected, 
            audio_token_shift=args.audio_token_shift,
            save_path=rejected_token_path
        )
    
    # 估算语音长度（基于字符数量）
    chosen_duration = len(normalized_text) * 0.2  # 每个字符约0.2秒
    rejected_duration = len(original_text) * 0.2
    
    # 创建DPO数据项
    return {
        "sample_id": sample_id,
        "original_text": original_text,
        "normalized_text": normalized_text,
        "chosen_token_path": chosen_token_path,
        "rejected_token_path": rejected_token_path,
        "chosen_wav_path": chosen_wav_path,
        "rejected_wav_path": rejected_wav_path,
        "chosen_duration": chosen_duration,
        "rejected_duration": rejected_duration
    }

def process_numerical_data(args, tokenizer, repcoder, prompt_text, prompt_speech_token, prompt_duration):
    """处理数值数据并创建DPO训练样本"""
    print(f"处理数值数据文件: {args.normalized_text_path}")
    
    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    
    # 读取normalized_text.json
    with open(args.normalized_text_path, 'r', encoding='utf-8') as f:
        numerical_data = json.load(f)
    
    # 限制样本数
    if args.max_samples > 0 and args.max_samples < len(numerical_data):
        numerical_data = numerical_data[:args.max_samples]
    
    # 随机打乱数据
    random.shuffle(numerical_data)
    
    # 划分训练集和测试集
    test_size = int(len(numerical_data) * args.test_split)
    train_data = numerical_data[test_size:]
    test_data = numerical_data[:test_size]
    
    print(f"数据集总量: {len(numerical_data)}")
    print(f"训练集大小: {len(train_data)} ({100 - args.test_split * 100:.0f}%)")
    print(f"测试集大小: {len(test_data)} ({args.test_split * 100:.0f}%)")
    
    # 保存测试集
    test_output_path = os.path.join(os.path.dirname(args.output_json), "test_data.json")
    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"测试集已保存至: {test_output_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存提示语音的token
    prompt_token_path = os.path.join(args.output_dir, "prompt_speech_token.npy")
    np.save(prompt_token_path, prompt_speech_token)
    
    # 准备并行处理的参数 - 只处理训练集
    item_args_list = [
        (idx, item, args.output_dir, args) 
        for idx, item in enumerate(train_data)
    ]
    
    # 并行处理样本
    all_results = []
    
    # 使用进程池处理样本
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_sample, item_args) for item_args in item_args_list]
        
        # 使用tqdm显示进度
        for future in tqdm(futures, total=len(futures), desc="处理样本"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"处理样本时出错: {e}")
    
    # 准备DPO数据
    dpo_data = []
    
    for result in all_results:
        # 创建DPO数据项
        dpo_item = {
            "prompt_text": prompt_text,
            "prompt_language": "zh",
            "prompt_wav_path": args.prompt_wav_path,
            "prompt_duration": prompt_duration,
            "prompt_speech_token_path": prompt_token_path,
            "target_text": result["normalized_text"],
            "target_language": "zh",
            "chosen": {
                "wav_path": result["chosen_wav_path"],
                "speech_token_path": result["chosen_token_path"],
                "duration": result["chosen_duration"]
            },
            "rejected": {
                "wav_path": result["rejected_wav_path"],
                "speech_token_path": result["rejected_token_path"],
                "duration": result["rejected_duration"]
            },
            "original_text": result["original_text"],
            "sample_id": result["sample_id"]
        }
        dpo_data.append(dpo_item)
    
    # 保存DPO数据
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，生成了 {len(dpo_data)} 个DPO训练样本")
    print(f"DPO数据已保存到：{args.output_json}")
    
    return dpo_data

def main():
    args = parse_args()
    
    # 加载模型
    tokenizer, repcoder = load_models(args)
    
    # 读取提示语音和文本
    prompt_text, prompt_speech_token, prompt_duration = read_prompt(
        args.prompt_wav_path, 
        args.prompt_text_path, 
        repcoder,
        args
    )
    
    # 处理数值数据
    process_numerical_data(
        args, 
        tokenizer, 
        repcoder, 
        prompt_text, 
        prompt_speech_token,
        prompt_duration
    )
    
    print("数据准备完成！")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main() 