#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class NumericalDPODataset(Dataset):
    """中文数字朗读DPO数据集"""
    
    def __init__(self, json_path, tokenizer=None, max_length=2048):
        """初始化数据集
        
        Args:
            json_path: DPO训练数据JSON文件路径
            tokenizer: 分词器，用于处理文本
            max_length: 序列最大长度
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载DPO训练数据"""
        print(f"加载数据: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 检查数据有效性
        valid_items = []
        for item in self.data:
            # 确保必要的字段存在
            if all(key in item for key in ["prompt_text", "target_text", "chosen", "rejected"]):
                # 确保speech_token_path_dict字段存在且文件存在
                chosen_token_path = item["chosen"]["speech_token_path_dict"]["extracted"]
                rejected_token_path = item["rejected"]["speech_token_path_dict"]["extracted"]
                
                if os.path.exists(chosen_token_path) and os.path.exists(rejected_token_path):
                    valid_items.append(item)
        
        self.data = valid_items
        print(f"有效样本数量: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取一个样本
        
        返回一个包含以下字段的字典:
            prompt_text: 提示文本
            target_text: 目标文本
            chosen_speech_tokens: 选择的语音tokens
            rejected_speech_tokens: 拒绝的语音tokens
        """
        item = self.data[idx]
        
        # 提取文本
        prompt_text = item["prompt_text"]
        target_text = item["target_text"]
        
        # 加载speech tokens
        chosen_token_path = item["chosen"]["speech_token_path_dict"]["extracted"]
        rejected_token_path = item["rejected"]["speech_token_path_dict"]["extracted"]
        
        chosen_speech_tokens = np.load(chosen_token_path)
        rejected_speech_tokens = np.load(rejected_token_path)
        
        # 加载prompt speech tokens (如果存在)
        prompt_speech_tokens = None
        if "prompt_speech_token_path" in item and os.path.exists(item["prompt_speech_token_path"]):
            prompt_speech_tokens = np.load(item["prompt_speech_token_path"])
        
        return {
            "prompt_text": prompt_text,
            "target_text": target_text,
            "chosen_speech_tokens": chosen_speech_tokens,
            "rejected_speech_tokens": rejected_speech_tokens,
            "prompt_speech_tokens": prompt_speech_tokens
        }
    
    @staticmethod
    def collate_fn(batch, tokenizer, audio_token_shift=32066, 
                   bos_audio_token_id=32064, eos_audio_token_id=32065):
        """数据批次整理函数
        
        处理一批样本，生成模型输入所需的格式
        """
        prompt_texts = [item["prompt_text"] + item["target_text"] for item in batch]
        
        # 使用tokenizer处理文本
        if tokenizer:
            # 获取text tokens
            text_encoding = tokenizer(prompt_texts, padding=True, return_tensors="pt")
            text_input_ids = text_encoding.input_ids
            text_attention_mask = text_encoding.attention_mask
        else:
            # 如果没有提供tokenizer，则只返回原始文本
            text_input_ids = prompt_texts
            text_attention_mask = None
        
        # 处理语音tokens
        chosen_tokens_list = [torch.tensor(item["chosen_speech_tokens"][0]) for item in batch]
        rejected_tokens_list = [torch.tensor(item["rejected_speech_tokens"][0]) for item in batch]
        
        # 为speech tokens添加特殊token
        processed_chosen_tokens = []
        processed_rejected_tokens = []
        
        for chosen, rejected in zip(chosen_tokens_list, rejected_tokens_list):
            # 添加bos和eos token
            chosen_with_special = torch.cat([
                torch.tensor([bos_audio_token_id]),
                chosen,
                torch.tensor([eos_audio_token_id])
            ])
            
            rejected_with_special = torch.cat([
                torch.tensor([bos_audio_token_id]),
                rejected,
                torch.tensor([eos_audio_token_id])
            ])
            
            processed_chosen_tokens.append(chosen_with_special)
            processed_rejected_tokens.append(rejected_with_special)
        
        # 填充到相同长度
        max_chosen_len = max(t.size(0) for t in processed_chosen_tokens)
        max_rejected_len = max(t.size(0) for t in processed_rejected_tokens)
        
        padded_chosen_tokens = torch.stack([
            torch.cat([t, torch.zeros(max_chosen_len - t.size(0), dtype=torch.long)]) 
            for t in processed_chosen_tokens
        ])
        
        padded_rejected_tokens = torch.stack([
            torch.cat([t, torch.zeros(max_rejected_len - t.size(0), dtype=torch.long)]) 
            for t in processed_rejected_tokens
        ])
        
        # 构建结果
        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "chosen_speech_tokens": padded_chosen_tokens,
            "rejected_speech_tokens": padded_rejected_tokens,
            "texts": prompt_texts  # 原始文本，用于调试
        }

def create_dpo_dataloader(json_path, tokenizer=None, batch_size=4, shuffle=True, num_workers=4):
    """创建DPO数据加载器
    
    Args:
        json_path: DPO训练数据JSON文件路径
        tokenizer: 分词器
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        
    Returns:
        DataLoader: DPO数据加载器
    """
    dataset = NumericalDPODataset(json_path, tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: NumericalDPODataset.collate_fn(batch, tokenizer)
    )
    
    return dataloader 