#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import datetime
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

# 添加项目根目录到 PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from numerical_dpo.models.dpo_model import DPOIntsModel
from numerical_dpo.data.dataset import NumericalDPODataset, create_dpo_dataloader

def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件
    log_file = os.path.join(log_dir, f"dpo_train_{timestamp}.log")
    
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("dpo_train")

def parse_args():
    parser = argparse.ArgumentParser(description="训练中文数字朗读DPO模型")
    parser.add_argument("--config_path", type=str, required=True, help="配置文件路径")
    parser.add_argument("--data_path", type=str, required=True, help="DPO数据文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--log_dir", type=str, default="numerical_dpo/logs", help="日志目录")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="学习率")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO温度参数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def prepare_batch_for_model(batch, model, device):
    """准备模型输入批次"""
    # 获取文本输入
    text_input_ids = batch["text_input_ids"].to(device)
    text_attention_mask = batch["text_attention_mask"].to(device)
    
    # 获取语音tokens
    chosen_speech_tokens = batch["chosen_speech_tokens"].to(device)
    rejected_speech_tokens = batch["rejected_speech_tokens"].to(device)
    
    # 构建标签 (只对语音部分计算损失)
    chosen_labels = -100 * torch.ones_like(text_input_ids)
    audio_mask = chosen_speech_tokens > model.audio_token_shift
    chosen_labels[audio_mask] = chosen_speech_tokens[audio_mask]
    
    rejected_labels = -100 * torch.ones_like(text_input_ids)
    audio_mask = rejected_speech_tokens > model.audio_token_shift
    rejected_labels[audio_mask] = rejected_speech_tokens[audio_mask]
    
    return {
        "chosen_input_ids": text_input_ids,
        "chosen_attention_mask": text_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": text_input_ids,  # 使用相同的文本输入
        "rejected_attention_mask": text_attention_mask,  # 使用相同的注意力掩码
        "rejected_labels": rejected_labels
    }

def train_epoch(model, dataloader, optimizer, scheduler, args, logger, epoch, step_offset=0):
    """训练一个epoch"""
    model.model.train()
    
    epoch_loss = 0
    epoch_chosen_rewards = 0
    epoch_rejected_rewards = 0
    
    # 创建进度条
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    # 梯度累积计数器
    accumulated_steps = 0
    
    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    for step, batch in enumerate(progress_bar):
        global_step = step + step_offset
        
        # 准备模型输入
        model_inputs = prepare_batch_for_model(batch, model, args.device)
        
        # 前向传播和损失计算
        if args.fp16:
            with torch.cuda.amp.autocast():
                loss, chosen_rewards, rejected_rewards = model.compute_dpo_loss(
                    chosen_input_ids=model_inputs["chosen_input_ids"],
                    chosen_attention_mask=model_inputs["chosen_attention_mask"],
                    chosen_labels=model_inputs["chosen_labels"],
                    rejected_input_ids=model_inputs["rejected_input_ids"],
                    rejected_attention_mask=model_inputs["rejected_attention_mask"],
                    rejected_labels=model_inputs["rejected_labels"],
                    beta=args.beta
                )
                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
        else:
            loss, chosen_rewards, rejected_rewards = model.compute_dpo_loss(
                chosen_input_ids=model_inputs["chosen_input_ids"],
                chosen_attention_mask=model_inputs["chosen_attention_mask"],
                chosen_labels=model_inputs["chosen_labels"],
                rejected_input_ids=model_inputs["rejected_input_ids"],
                rejected_attention_mask=model_inputs["rejected_attention_mask"],
                rejected_labels=model_inputs["rejected_labels"],
                beta=args.beta
            )
            # 梯度累积
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
        
        accumulated_steps += 1
        
        # 如果达到梯度累积步数，则更新参数
        if accumulated_steps >= args.gradient_accumulation_steps:
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            accumulated_steps = 0
        
        # 累积指标
        epoch_loss += loss.item() * args.gradient_accumulation_steps
        epoch_chosen_rewards += chosen_rewards.item()
        epoch_rejected_rewards += rejected_rewards.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            "loss": loss.item() * args.gradient_accumulation_steps,
            "chosen_reward": chosen_rewards.item(),
            "rejected_reward": rejected_rewards.item()
        })
        
        # 打印日志
        if global_step % args.logging_steps == 0:
            logger.info(f"Epoch: {epoch+1}, Step: {global_step}, Loss: {loss.item() * args.gradient_accumulation_steps:.6f}, " +
                       f"Chosen Reward: {chosen_rewards.item():.6f}, Rejected Reward: {rejected_rewards.item():.6f}")
        
        # 保存模型
        if global_step > 0 and global_step % args.save_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            model.save_model(checkpoint_dir)
            logger.info(f"保存模型到: {checkpoint_dir}")
    
    # 计算平均指标
    avg_loss = epoch_loss / len(dataloader)
    avg_chosen_rewards = epoch_chosen_rewards / len(dataloader)
    avg_rejected_rewards = epoch_rejected_rewards / len(dataloader)
    
    logger.info(f"Epoch {epoch+1} 训练完成，平均损失: {avg_loss:.6f}, " +
               f"平均Chosen奖励: {avg_chosen_rewards:.6f}, 平均Rejected奖励: {avg_rejected_rewards:.6f}")
    
    return global_step + 1

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info(f"开始训练DPO模型，参数: {args}")
    
    # 加载配置文件
    config = load_config(args.config_path)
    logger.info(f"加载配置文件: {args.config_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    logger.info(f"初始化模型: {args.model_path}")
    model = DPOIntsModel(
        model_path=args.model_path,
        device=args.device
    )
    
    # 加载数据集
    logger.info(f"加载数据集: {args.data_path}")
    dpo_dataloader = create_dpo_dataloader(
        json_path=args.data_path,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 准备优化器
    optimizer = AdamW(model.model.parameters(), lr=args.learning_rate)
    
    # 准备学习率调度器
    num_training_steps = args.num_epochs * len(dpo_dataloader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # 训练模型
    logger.info("开始训练...")
    step_offset = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"开始第 {epoch+1}/{args.num_epochs} 轮训练")
        step_offset = train_epoch(
            model=model, 
            dataloader=dpo_dataloader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            args=args,
            logger=logger,
            epoch=epoch,
            step_offset=step_offset
        )
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_model(final_model_path)
    logger.info(f"训练完成，保存最终模型到: {final_model_path}")

if __name__ == "__main__":
    main() 