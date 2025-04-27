#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并g2pm/data目录下的训练集、开发集和测试集数据
将所有的.sent文件合并为data.sent
将所有的.lb文件合并为data.lb
并保存到polyphone目录下
"""

import os
import sys

def merge_files(input_files, output_file):
    """
    合并多个文件内容到一个文件中
    
    参数:
        input_files: 需要合并的输入文件列表
        output_file: 合并后的输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as outf:
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as inf:
                    content = inf.read()
                    outf.write(content)
                    # 如果文件最后一行没有换行符，则添加一个
                    if content and not content.endswith('\n'):
                        outf.write('\n')
                print(f"已合并文件: {file_path}")
            except Exception as e:
                print(f"合并文件 {file_path} 时出错: {e}")

def main():
    # 确保输出目录存在
    output_dir = "polyphone"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输入和输出文件
    data_dir = "g2pm/data"
    # 所有需要合并的sent文件路径列表
    sent_files = [
        os.path.join(data_dir, "train.sent"),  # 训练集句子文件
        os.path.join(data_dir, "dev.sent"),    # 开发集句子文件
        os.path.join(data_dir, "test.sent")    # 测试集句子文件
    ]
    # 所有需要合并的lb文件路径列表
    lb_files = [
        os.path.join(data_dir, "train.lb"),    # 训练集标签文件
        os.path.join(data_dir, "dev.lb"),      # 开发集标签文件
        os.path.join(data_dir, "test.lb")      # 测试集标签文件
    ]
    
    # 合并后的输出文件路径
    output_sent = os.path.join(output_dir, "data.sent")
    output_lb = os.path.join(output_dir, "data.lb")
    
    # 合并sent文件
    print("开始合并sent文件...")
    merge_files(sent_files, output_sent)
    
    # 合并lb文件
    print("开始合并lb文件...")
    merge_files(lb_files, output_lb)
    
    # 检查合并结果
    try:
        # 统计合并后文件的行数
        sent_count = sum(1 for _ in open(output_sent, 'r', encoding='utf-8'))
        lb_count = sum(1 for _ in open(output_lb, 'r', encoding='utf-8'))
        print(f"合并完成！data.sent文件包含{sent_count}行，data.lb文件包含{lb_count}行")
        
        # 检查两个文件的行数是否一致，一致性验证
        if sent_count != lb_count:
            print(f"警告：合并后的sent文件行数({sent_count})与lb文件行数({lb_count})不一致！")
        else:
            print("验证通过：sent文件与lb文件行数一致，合并成功！")
    except Exception as e:
        print(f"检查合并结果时出错: {e}")

if __name__ == "__main__":
    main() 