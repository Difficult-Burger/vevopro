#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import re
import sys

class NumericalDataProcessor:
    """数值数据处理类，用于处理normalized_text.json数据"""
    
    def __init__(self, normalized_text_path, output_dir, max_samples=None):
        """初始化数据处理器
        
        Args:
            normalized_text_path: normalized_text.json路径
            output_dir: 输出目录
            max_samples: 最大样本数，None表示处理所有样本
        """
        self.normalized_text_path = normalized_text_path
        self.output_dir = output_dir
        self.max_samples = max_samples
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载normalized_text.json数据"""
        print(f"加载数据: {self.normalized_text_path}")
        with open(self.normalized_text_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 限制样本数
        if self.max_samples is not None and self.max_samples < len(self.data):
            self.data = self.data[:self.max_samples]
            print(f"限制样本数量为: {self.max_samples}")
        
        print(f"加载了 {len(self.data)} 个样本")
    
    def extract_number_patterns(self):
        """从数据中提取数字模式"""
        patterns = []
        
        for item in self.data:
            original = item["original"]
            normalized = item["normalized"]
            
            # 使用正则表达式查找数字pattern
            # 查找数字和常见的分隔符（如:/-等）
            digit_patterns = re.findall(r'[0-9:\/\-\.]+', original)
            
            if digit_patterns:
                for pattern in digit_patterns:
                    patterns.append({
                        "original": pattern,
                        "original_context": original,
                        "normalized_context": normalized
                    })
        
        # 输出结果
        patterns_file = os.path.join(self.output_dir, "number_patterns.json")
        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
        
        print(f"提取了 {len(patterns)} 个数字模式，保存至: {patterns_file}")
        return patterns
    
    def create_dpo_data_json(self, prompt_text, prompt_speech_token_path, prompt_duration):
        """创建DPO训练数据JSON文件
        
        Args:
            prompt_text: 提示文本
            prompt_speech_token_path: 提示语音token路径
            prompt_duration: 提示语音时长（秒）
            
        Returns:
            list: DPO训练数据
        """
        dpo_data = []
        
        for idx, item in enumerate(tqdm(self.data, desc="创建DPO数据")):
            original_text = item["original"]
            normalized_text = item["normalized"]
            
            # 为每个样本创建唯一ID
            sample_id = f"numerical_{idx:06d}"
            
            # 创建子目录
            chosen_dir = os.path.join(self.output_dir, f"{sample_id}_chosen")
            rejected_dir = os.path.join(self.output_dir, f"{sample_id}_rejected")
            os.makedirs(chosen_dir, exist_ok=True)
            os.makedirs(rejected_dir, exist_ok=True)
            
            # token文件路径
            chosen_token_path = os.path.join(chosen_dir, f"{sample_id}_chosen.npy")
            rejected_token_path = os.path.join(rejected_dir, f"{sample_id}_rejected.npy")
            
            # 音频文件路径
            chosen_wav_path = os.path.join(chosen_dir, f"{sample_id}_chosen.wav")
            rejected_wav_path = os.path.join(rejected_dir, f"{sample_id}_rejected.wav")
            
            # 估算语音长度（基于字符数量）
            chosen_duration = len(normalized_text) * 0.2  # 每个字符约0.2秒
            rejected_duration = len(original_text) * 0.2  # 每个字符约0.2秒
            
            # 创建DPO数据项
            dpo_item = {
                "prompt_text": prompt_text,
                "prompt_language": "zh",
                "prompt_wav_path": "",  # 实际使用时填充
                "prompt_duration": prompt_duration,
                "prompt_speech_token_path": prompt_speech_token_path,
                "target_text": normalized_text,
                "target_language": "zh",
                "chosen": {
                    "wav_path": chosen_wav_path,
                    "duration": chosen_duration,
                    "speech_token_path_dict": {
                        "extracted": chosen_token_path,
                        "predicted": chosen_token_path
                    }
                },
                "rejected": {
                    "wav_path": rejected_wav_path,
                    "duration": rejected_duration,
                    "speech_token_path_dict": {
                        "extracted": rejected_token_path,
                        "predicted": rejected_token_path
                    }
                }
            }
            
            dpo_data.append(dpo_item)
        
        return dpo_data
    
    @staticmethod
    def generate_random_tokens(length, audio_token_shift=32066, save_path=None):
        """生成随机语音tokens（用于测试）
        
        Args:
            length: token序列长度
            audio_token_shift: 音频token偏移值
            save_path: 保存路径，None表示不保存
            
        Returns:
            np.ndarray: 随机tokens
        """
        # 生成1000到5000之间的随机值
        random_tokens = np.random.randint(1000, 5000, size=(1, length))
        
        # 应用audio_token_shift
        shifted_tokens = random_tokens + audio_token_shift
        
        # 保存到文件
        if save_path is not None:
            np.save(save_path, shifted_tokens)
        
        return shifted_tokens
    
    def create_mock_dpo_data(self):
        """创建模拟DPO训练数据（用于测试）"""
        # 创建提示语音token
        prompt_token_path = os.path.join(self.output_dir, "prompt_speech_token.npy")
        prompt_tokens = self.generate_random_tokens(200, save_path=prompt_token_path)
        prompt_duration = len(prompt_tokens[0]) * 0.02  # 每个token约0.02秒
        
        # 创建模拟提示文本
        prompt_text = "请阅读以下文本"
        
        # 创建DPO数据
        dpo_data = self.create_dpo_data_json(prompt_text, prompt_token_path, prompt_duration)
        
        # 为每个样本生成mock token文件
        for item in tqdm(dpo_data, desc="生成模拟token"):
            # 生成chosen语音tokens
            chosen_token_path = item["chosen"]["speech_token_path_dict"]["extracted"]
            chosen_length = int(item["chosen"]["duration"] / 0.02)  # 每个token约0.02秒
            self.generate_random_tokens(chosen_length, save_path=chosen_token_path)
            
            # 生成rejected语音tokens
            rejected_token_path = item["rejected"]["speech_token_path_dict"]["extracted"]
            rejected_length = int(item["rejected"]["duration"] / 0.02)  # 每个token约0.02秒
            self.generate_random_tokens(rejected_length, save_path=rejected_token_path)
        
        # 保存DPO数据
        dpo_data_path = os.path.join(self.output_dir, "dpo_data.json")
        with open(dpo_data_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        
        print(f"创建了 {len(dpo_data)} 个模拟DPO训练样本，保存至: {dpo_data_path}")
        return dpo_data_path

class RepCoderProcessor:
    """RepCoder语音编码处理类"""
    
    def __init__(self, repcoder, device="cuda:0"):
        """初始化RepCoder处理器
        
        Args:
            repcoder: RepCoder实例
            device: 设备
        """
        self.repcoder = repcoder
        self.device = device
    
    def extract_speech_tokens(self, audio_path, target_sample_rate=16000):
        """从音频文件提取语音tokens
        
        Args:
            audio_path: 音频文件路径
            target_sample_rate: 目标采样率
            
        Returns:
            np.ndarray: 语音tokens
        """
        # 读取音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 确保是单声道
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 转换采样率
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        
        # 提取语音token
        return self.extract_speech_tokens_from_waveform(waveform)
    
    def extract_speech_tokens_from_waveform(self, waveform):
        """从波形数据提取语音tokens
        
        Args:
            waveform: 波形数据
            
        Returns:
            np.ndarray: 语音tokens
        """
        # 将波形转到设备上
        waveform_cuda = waveform.to(self.device)
        
        # 提取语音token
        with torch.no_grad():
            semantic_vectors = self.repcoder.extract_semantic_vectors(waveform_cuda)
            speech_token = self.repcoder.quantize_by_kmeans(semantic_vectors)
        
        return speech_token.cpu().numpy()
    
    def batch_process_audio_files(self, audio_files, output_dir, target_sample_rate=16000):
        """批量处理音频文件，提取语音tokens
        
        Args:
            audio_files: 音频文件路径列表
            output_dir: 输出目录
            target_sample_rate: 目标采样率
            
        Returns:
            list: 处理结果
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            # 构建输出路径
            base_name = os.path.basename(audio_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}.npy")
            
            # 提取语音tokens
            speech_tokens = self.extract_speech_tokens(audio_file, target_sample_rate)
            
            # 保存tokens
            np.save(output_path, speech_tokens)
            
            # 记录结果
            results.append({
                "audio_file": audio_file,
                "token_file": output_path,
                "token_length": len(speech_tokens[0])
            })
        
        # 保存处理结果
        results_file = os.path.join(output_dir, "processing_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"处理了 {len(results)} 个音频文件，结果保存至: {results_file}")
        return results 