#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到 PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def parse_args():
    parser = argparse.ArgumentParser(description="从WAV音频文件中提取语音tokens")
    parser.add_argument("--w2v_bert_path", type=str, required=True, help="wav2vec-BERT模型路径")
    parser.add_argument("--dual_codec_path", type=str, required=True, help="RepCodec路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入音频目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--sample_rate", type=int, default=16000, help="目标采样率")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 添加RepCodec路径到系统路径
    sys.path.append(args.dual_codec_path)
    
    # 导入RepCoder
    from RepCodec.RepCoder import RepCoder
    
    # 加载RepCoder
    print(f"加载RepCoder模型: {args.w2v_bert_path}")
    repcoder = RepCoder(
        path=args.w2v_bert_path,
        device=args.device
    )
    
    # 导入RepCoderProcessor
    from numerical_dpo.utils.data_processor import RepCoderProcessor
    
    # 创建RepCoderProcessor
    processor = RepCoderProcessor(repcoder, device=args.device)
    
    # 查找输入目录中的所有WAV文件
    input_dir = Path(args.input_dir)
    audio_files = list(input_dir.glob("**/*.wav"))
    
    if not audio_files:
        print(f"在目录 {args.input_dir} 中未找到WAV文件！")
        return
    
    print(f"找到 {len(audio_files)} 个WAV文件")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 批量处理音频文件
    results = processor.batch_process_audio_files(
        audio_files=[str(f) for f in audio_files],
        output_dir=args.output_dir,
        target_sample_rate=args.sample_rate
    )
    
    # 打印处理结果
    print("\n处理完成！")
    print(f"已将 {len(results)} 个音频文件的语音tokens提取到 {args.output_dir} 目录")
    
    # 保存输入和输出的映射关系
    mapping_file = os.path.join(args.output_dir, "audio_token_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            "audio_files": [str(f) for f in audio_files],
            "token_files": [r["token_file"] for r in results],
            "token_lengths": [r["token_length"] for r in results]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"映射关系已保存至: {mapping_file}")

if __name__ == "__main__":
    main() 