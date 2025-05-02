#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F

class DPOIntsModel:
    """DPO训练用的Ints TTS模型封装"""
    
    def __init__(self, model_path, tokenizer_path=None, device="cuda:0"):
        """初始化DPO模型
        
        Args:
            model_path: 模型路径
            tokenizer_path: 分词器路径，默认与model_path相同
            device: 设备
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path else model_path
        self.device = device
        
        # 加载tokenizer
        print(f"加载tokenizer: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # 加载模型
        print(f"加载模型: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 设置特殊token ID
        self.bos_audio_token_id = 32064
        self.eos_audio_token_id = 32065
        self.end_token_id = 2
        self.audio_token_shift = 32066
        
        # 初始化参考模型（用于DPO训练）
        self.reference_model = None
    
    def load_reference_model(self):
        """加载参考模型（用于DPO训练）"""
        if self.reference_model is None:
            print(f"加载参考模型: {self.model_path}")
            self.reference_model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.reference_model.to(self.device)
            self.reference_model.eval()
            
            # 冻结参考模型参数
            for param in self.reference_model.parameters():
                param.requires_grad = False
    
    def gen_chat_prompt_from_text(self, text):
        """生成聊天格式的提示"""
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def generate_speech_tokens(self, text, max_new_tokens=3000, top_p=0.9, temperature=1.0):
        """从文本生成语音tokens
        
        Args:
            text: 输入文本
            max_new_tokens: 最大生成token数
            top_p: 采样参数top_p
            temperature: 采样温度
            
        Returns:
            np.ndarray: 生成的语音tokens
        """
        # 生成聊天格式的提示
        chat_prompt = self.gen_chat_prompt_from_text(text)
        
        # 编码输入
        input_ids = self.tokenizer(chat_prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 生成语音tokens
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 提取生成的语音tokens
        generated_ids = outputs[0].cpu().numpy()
        
        # 找到语音tokens区域
        # 从第一个bos_audio_token_id到最后
        audio_start_idx = np.where(generated_ids == self.bos_audio_token_id)[0]
        if len(audio_start_idx) == 0:
            print(f"WARNING: 未生成语音tokens，文本: {text}")
            return np.array([])
        
        audio_start_idx = audio_start_idx[-1]  # 取最后一个bos_audio_token_id
        audio_tokens = generated_ids[audio_start_idx:]
        
        return audio_tokens
    
    def compute_logprobs(self, input_ids, attention_mask, labels):
        """计算语音tokens的对数概率
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            labels: 标签
            
        Returns:
            torch.Tensor: 对数概率
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            logits = outputs.logits
            
            # 只计算有效的token (非padding)
            valid_mask = (labels != -100).float()
            
            # 计算对数概率
            log_probs = torch.gather(
                F.log_softmax(logits, dim=-1),
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # 应用掩码并计算和
            masked_log_probs = log_probs * valid_mask
            sum_log_probs = masked_log_probs.sum(dim=-1)
            
            return sum_log_probs
    
    def compute_dpo_loss(self, chosen_input_ids, chosen_attention_mask, chosen_labels,
                        rejected_input_ids, rejected_attention_mask, rejected_labels,
                        beta=0.1):
        """计算DPO损失
        
        Args:
            chosen_input_ids: 选择的输入ID
            chosen_attention_mask: 选择的注意力掩码
            chosen_labels: 选择的标签
            rejected_input_ids: 拒绝的输入ID
            rejected_attention_mask: 拒绝的注意力掩码
            rejected_labels: 拒绝的标签
            beta: DPO温度参数
            
        Returns:
            torch.Tensor: DPO损失
        """
        # 确保参考模型已加载
        if self.reference_model is None:
            self.load_reference_model()
        
        # 计算策略模型的对数概率
        policy_chosen_logps = self.compute_logprobs(chosen_input_ids, chosen_attention_mask, chosen_labels)
        policy_rejected_logps = self.compute_logprobs(rejected_input_ids, rejected_attention_mask, rejected_labels)
        
        # 计算参考模型的对数概率
        with torch.no_grad():
            ref_chosen_logps = self.compute_logprobs(chosen_input_ids, chosen_attention_mask, chosen_labels)
            ref_rejected_logps = self.compute_logprobs(rejected_input_ids, rejected_attention_mask, rejected_labels)
        
        # 计算log-ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # 计算DPO logits
        logits = chosen_logratios - rejected_logratios
        
        # 使用sigmoid损失
        losses = -F.logsigmoid(beta * logits)
        
        # 计算奖励值(用于监控)
        chosen_rewards = (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - ref_rejected_logps).detach()
        
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
    
    def save_model(self, output_path):
        """保存模型
        
        Args:
            output_path: 输出路径
        """
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_path)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_path)
        
        print(f"模型已保存至: {output_path}")
    
    def batch_generate_speech_tokens(self, texts, max_new_tokens=3000, top_p=0.9, temperature=1.0):
        """批量生成语音tokens
        
        Args:
            texts: 输入文本列表
            max_new_tokens: 最大生成token数
            top_p: 采样参数top_p
            temperature: 采样温度
            
        Returns:
            list: 生成的语音tokens列表
        """
        results = []
        for text in tqdm(texts, desc="生成语音tokens"):
            tokens = self.generate_speech_tokens(
                text, 
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature
            )
            results.append(tokens)
        return results
    
    def create_numerical_dpo_samples(self, original_texts, normalized_texts, max_new_tokens=3000):
        """为数值DPO训练创建对比样本
        
        Args:
            original_texts: 原始文本列表(包含数字)
            normalized_texts: 标准化文本列表(数字已标准化)
            max_new_tokens: 最大生成token数
            
        Returns:
            dict: 包含生成的对比样本信息
        """
        assert len(original_texts) == len(normalized_texts), "原始文本和标准化文本数量不匹配"
        
        results = []
        
        for idx, (orig, norm) in enumerate(zip(original_texts, normalized_texts)):
            print(f"处理样本 {idx+1}/{len(original_texts)}")
            
            # 生成原始文本的语音tokens (rejected)
            rejected_tokens = self.generate_speech_tokens(orig, max_new_tokens=max_new_tokens)
            
            # 生成标准化文本的语音tokens (chosen)
            chosen_tokens = self.generate_speech_tokens(norm, max_new_tokens=max_new_tokens)
            
            if len(rejected_tokens) == 0 or len(chosen_tokens) == 0:
                print(f"跳过样本 {idx+1}，生成失败")
                continue
            
            # 创建样本
            sample = {
                "original_text": orig,
                "normalized_text": norm,
                "chosen_speech_tokens": chosen_tokens,
                "rejected_speech_tokens": rejected_tokens
            }
            
            results.append(sample)
        
        return {
            "samples": results,
            "count": len(results),
            "original_count": len(original_texts)
        } 