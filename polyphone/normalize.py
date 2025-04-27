import json
import re
import time
from openai import OpenAI
import os
from tqdm import tqdm
import sys

# 添加上级目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入config模块
from numerical.config import BASE_URL, API_KEY

# 初始化OpenAI客户端 - 连接DeepSeek API
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def call_deepseek_api(text, polyphone, pronunciation=None):
    """
    调用DeepSeek API查找多音字的同音且唯一读音替代字
    
    参数:
    text: 包含多音字的原始文本
    polyphone: 需要替换的多音字
    pronunciation: 多音字的读音（拼音+声调）
    
    返回:
    替换后的字符
    """
    # 如果是轻声(音调5)，直接返回原字
    if pronunciation and pronunciation.endswith('5'):
        print(f"保留轻声字: {polyphone} ({pronunciation})")
        return polyphone
    
    # 提取多音字上下文，用于确定读音
    context = text
    
    # 设计prompt，指导模型查找同音唯一读音的汉字
    prompt = f"""请帮我找到一个与"{polyphone}"在当前上下文中发音相同，但只有唯一一种发音的汉字。
    
上下文句子: {context}

"""
    
    # 如果有提供读音信息，加入到prompt中
    if pronunciation:
        prompt += f'多音字"{polyphone}"在此句中的准确读音是：{pronunciation}。\n\n'
        prompt += f'请找一个与"{polyphone}"发音为"{pronunciation}"相同，但只有唯一一种读音的汉字作为替代。\n\n'
    else:
        prompt += f'在这个上下文中，请首先确定"{polyphone}"的准确读音（拼音），然后找一个只有这一种读音的其他汉字作为替代。\n\n'
    
    # 添加忽略语义问题的指示和三引号格式要求
    prompt += """请注意：直接忽略替换后可能出现的语义问题，只关注发音的一致性和替代字是否只有唯一读音。

请按以下格式回答，并将最终替代字放在三引号内：
原字: [原多音字]
当前读音: [拼音]
替代字: 
'''[替代汉字]'''
替代字拼音: [拼音]
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个精通汉语发音和多音字的专家助手。你的任务是帮助找到只有唯一读音的汉字来替代多音字，以便于文本转语音系统准确发音。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            stream=False
        )
        
        response_text = response.choices[0].message.content
        print(f"原字: {polyphone}, 读音: {pronunciation}, API响应: {response_text}")
        
        # 从回答中提取替代字（适应三引号格式）
        substitute_match = re.search(r"'''([^']+)'''", response_text)
        if substitute_match:
            substitute = substitute_match.group(1)
            return substitute
        else:
            # 尝试其他提取方式（兼容旧格式）
            substitute_match = re.search(r'替代字:\s*(\S+)', response_text)
            if substitute_match:
                substitute = substitute_match.group(1)
                return substitute
            
            # 如果没有找到标准格式的替代字，尝试其他提取方法
            # 先查找所有中文字符
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', response_text)
            # 过滤掉原多音字
            alternative_chars = [char for char in chinese_chars if char != polyphone]
            if alternative_chars:
                # 返回找到的第一个可能的替代字
                return alternative_chars[0]
            
            # 如果实在找不到，返回原字
            return polyphone
    except Exception as e:
        print(f"API调用错误: {e}")
        return polyphone

def process_file(input_file, pronunciation_file, output_file, max_count=None):
    """
    处理文件，将多音字替换为同音唯一读音的字
    
    参数:
    input_file: 输入文件路径
    pronunciation_file: 多音字读音文件路径
    output_file: 输出文件路径
    max_count: 最大处理数据条数，None表示处理全部
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 读取读音文件
    with open(pronunciation_file, 'r', encoding='utf-8') as f:
        pronunciations = f.readlines()
    
    # 确保文本行数与读音行数一致
    if len(lines) != len(pronunciations):
        raise ValueError(f"文本行数({len(lines)})与读音行数({len(pronunciations)})不一致")
    
    # 设置正则表达式模式，查找被下划线标记的多音字
    pattern = r'▁(.+?)▁'
    
    # 计数变量
    cnt = 0
    
    # 初始化JSON文件，先写入开始的方括号
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    # 遍历每行文本
    for i, line in enumerate(tqdm(lines, desc="处理句子")):
        line = line.strip()
        if not line:
            continue
        
        # 获取对应的读音行
        pronunciation_line = pronunciations[i].strip()
        
        # 查找所有多音字
        original_text = line
        modified_text = line
        polyphonic_chars = re.findall(pattern, line)
        
        # 如果有多音字，处理该行
        if polyphonic_chars:
            # 拆分读音行得到每个多音字的读音
            pronunciation_parts = pronunciation_line.split()
            
            # 确保多音字个数与读音个数一致
            if len(polyphonic_chars) == len(pronunciation_parts):
                # 替换每个多音字
                for j, polyphone in enumerate(polyphonic_chars):
                    # 获取对应读音
                    pron = pronunciation_parts[j]
                    
                    # 调用API查找替代字或保留原字（轻声）
                    substitute = call_deepseek_api(original_text, polyphone, pron)
                    
                    # 替换文本中的多音字
                    modified_text = modified_text.replace(f"▁{polyphone}▁", substitute, 1)
                    
                    # 添加短暂延迟，避免API限制
                    time.sleep(0.01)
            else:
                print(f"警告: 行 {i+1} 的多音字个数({len(polyphonic_chars)})与读音个数({len(pronunciation_parts)})不匹配")
                # 当无法匹配读音时，不使用读音信息
                for polyphone in polyphonic_chars:
                    # 调用API查找替代字
                    substitute = call_deepseek_api(original_text, polyphone)
                    
                    # 替换文本中的多音字
                    modified_text = modified_text.replace(f"▁{polyphone}▁", substitute, 1)
                    
                    # 添加短暂延迟，避免API限制
                    time.sleep(0.001)
        
        # 添加到JSON文件
        result = {
            "original": original_text,
            "normalized": modified_text
        }
        
        # 将结果立即追加到文件
        with open(output_file, 'a', encoding='utf-8') as f:
            # 处理不是第一条记录时需要添加的逗号
            if cnt > 0:
                f.write(',\n')
            
            # 写入当前结果
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 增加计数
        cnt += 1
        
        # 如果达到指定数量，停止处理
        if max_count and cnt >= max_count:
            print(f"已达到指定处理数量 {max_count}，停止处理")
            break
    
    # 完成处理后，添加结束的方括号
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    print(f"处理完成，共转换 {cnt} 条记录，已保存到 {output_file}")

if __name__ == "__main__":
    input_file = "polyphone/data.sent"
    pronunciation_file = "polyphone/data.lb"
    output_file = "polyphone/normalized.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 设置临时处理数量限制
    max_count = 200  # 只处理30条数据进行测试
    
    # 处理文件
    process_file(input_file, pronunciation_file, output_file, max_count)
