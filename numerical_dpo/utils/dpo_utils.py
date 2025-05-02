#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import torchaudio
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def gen_chat_prompt_from_text_phi3(text):
    """生成Phi-3格式的聊天提示"""
    prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def extract_speech_tokens_from_audio(audio_path, repcoder, target_sample_rate=16000):
    """从音频文件中提取语音tokens"""
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
    waveform_cuda = waveform.to(repcoder.device)
    with torch.no_grad():
        semantic_vectors = repcoder.extract_semantic_vectors(waveform_cuda)
        speech_token = repcoder.quantize_by_kmeans(semantic_vectors)
    
    return speech_token.cpu().numpy()

def load_ints_model(model_path, device="cuda:0"):
    """加载Ints模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer

def normalize_chinese_numbers(text):
    """将中文数字标准化处理
    
    例如：
    1. 将阿拉伯数字转换为中文数字表示
    2. 标准化数字读法格式
    """
    # 简单的阿拉伯数字到中文的映射示例
    # 实际应用中应使用更复杂的逻辑处理各种情况
    arab_to_cn = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    
    for arab, cn in arab_to_cn.items():
        text = text.replace(arab, cn)
    
    # 这里应添加更多规则来处理复杂情况
    # 如：日期、时间、分数、小数等
    
    return text

def process_paired_samples(original_texts, normalized_texts, model, tokenizer, repcoder, 
                         audio_token_shift=32066, bos_audio_token_id=32064, 
                         eos_audio_token_id=32065, device="cuda:0"):
    """处理成对的原始文本和标准化文本，生成DPO训练数据对"""
    results = []
    
    for orig, norm in zip(original_texts, normalized_texts):
        # 对文本进行编码
        orig_prompt = gen_chat_prompt_from_text_phi3(orig)
        norm_prompt = gen_chat_prompt_from_text_phi3(norm)
        
        orig_inputs = tokenizer(orig_prompt, return_tensors="pt").to(device)
        norm_inputs = tokenizer(norm_prompt, return_tensors="pt").to(device)
        
        # 在实际应用中，这里应该使用模型生成语音tokens
        # 这里只是示例，实际需要根据模型生成过程修改
        with torch.no_grad():
            # 这里只是模拟生成tokens的过程
            # 实际应通过模型生成
            chosen_speech_token = torch.randint(0, 1000, (1, len(norm) * 5)).numpy()
            rejected_speech_token = torch.randint(0, 1000, (1, len(orig) * 5)).numpy()
        
        # 构建DPO样本对
        sample_pair = {
            "chosen": chosen_speech_token + audio_token_shift,
            "rejected": rejected_speech_token + audio_token_shift,
            "original_text": orig,
            "normalized_text": norm
        }
        
        results.append(sample_pair)
    
    return results

def prepare_dpo_batch(text_tokens, speech_tokens, attention_mask=None, 
                    loss_mask=None, labels=None, audio_token_shift=32066):
    """准备DPO批次数据"""
    if attention_mask is None:
        attention_mask = torch.ones_like(text_tokens)
    
    if loss_mask is None:
        # 只对speech_tokens部分计算损失
        loss_mask = torch.zeros_like(text_tokens)
        loss_mask[text_tokens >= audio_token_shift] = 1
    
    if labels is None:
        # 只对speech_tokens部分计算损失
        labels = -100 * torch.ones_like(text_tokens)
        speech_mask = text_tokens >= audio_token_shift
        labels[speech_mask] = text_tokens[speech_mask]
    
    return {
        "input_ids": text_tokens,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "labels": labels
    } 