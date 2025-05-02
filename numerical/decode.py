#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from tqdm import tqdm

def decode_jsonl_file(input_file, output_file):
    """
    从JSONL文件中解码文本数据并保存到新的JSON文件中
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSON文件路径
    """
    # 用于存储解码后的数据
    decoded_data = []
    
    # 读取并解码JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        # 使用tqdm显示进度
        for line in tqdm(f, desc="解码数据"):
            try:
                # 解析JSON行
                item = json.loads(line.strip())
                
                # 提取id和文本
                decoded_item = {
                    "id": item["id"],
                    "text": item["text"]  # 文本已经是Unicode字符串，不需要额外解码
                }
                
                # 添加到结果列表
                decoded_data.append(decoded_item)
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line[:50]}...")
            except Exception as e:
                print(f"处理数据时出错: {e}")
    
    # 将解码后的数据保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(decoded_data, f, ensure_ascii=False, indent=2)
    
    print(f"已成功解码 {len(decoded_data)} 条数据并保存到 {output_file}")

if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入和输出文件路径
    input_file = os.path.join(current_dir, "CN_TN_epoch-01-28645_2.jsonl")
    output_file = os.path.join(current_dir, "decoded_text.json")
    
    # 解码数据
    decode_jsonl_file(input_file, output_file) 