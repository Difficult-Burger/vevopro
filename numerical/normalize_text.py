import json
import time
import os
import re
from tqdm import tqdm
from openai import OpenAI
from config import BASE_URL, API_KEY

# 初始化OpenAI客户端 - 新的调用方式
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 原来的客户端初始化方式
# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 处理日期格式，去除月份和日期的前导零
def process_date_format(text):
    # 匹配年月日格式
    date_pattern = re.compile(r'(\d{4})年0?(\d{1,2})月0?(\d{1,2})日')
    # 匹配月日格式
    month_day_pattern = re.compile(r'0?(\d{1,2})月0?(\d{1,2})日')
    # 匹配简写年月格式
    year_month_pattern = re.compile(r'(\d{2})年0?(\d{1,2})月')
    
    # 年月日格式替换
    text = date_pattern.sub(r'\1年\2月\3日', text)
    # 月日格式替换
    text = month_day_pattern.sub(r'\1月\2日', text)
    # 简写年月格式替换
    text = year_month_pattern.sub(r'\1年\2月', text)
    
    return text

# 处理时间格式，去除小时前导零
def process_time_format(text):
    # 匹配时间格式 hh:mm
    time_pattern = re.compile(r'(\d{2}):(\d{2})')
    
    def time_replace(match):
        original_hour = match.group(1)
        hour = original_hour.lstrip('0') or '0'  # 去除小时前导零，但保留单独的'0'
        minute = match.group(2)
        return f"{hour}点{minute}分"
            
    result = time_pattern.sub(time_replace, text)
    return result

# 处理温度格式，将"xx度C"转换为"xx摄氏度"，将"xx度F"转换为"xx华氏度"
def process_temperature_format(text):
    # 匹配摄氏度格式 如 25度C、37.5度C
    celsius_pattern = re.compile(r'(\d+(?:\.\d+)?)度C')
    text = celsius_pattern.sub(r'\1摄氏度', text)
    
    # 匹配华氏度格式 如 98度F、99.5度F
    fahrenheit_pattern = re.compile(r'(\d+(?:\.\d+)?)度F')
    text = fahrenheit_pattern.sub(r'\1华氏度', text)
    
    return text

# 处理HTML标签，将其删除
def remove_html_tags(text):
    """删除文本中的HTML标签"""
    # 使用正则表达式去除HTML标签
    clean_text = re.sub(r'<[^>]*>', '', text)
    
    return clean_text

# 预处理文本，处理日期和时间格式
def preprocess_text(text):
    # 去除HTML标签
    text = remove_html_tags(text)
    # 处理温度格式
    text = process_temperature_format(text)
    # 处理时间格式
    text = process_time_format(text)
    # 处理日期格式
    text = process_date_format(text)
    return text

def call_deepseek_api(text):
    """调用DeepSeek API将数字转为汉字表达"""
    # 首先预处理文本，处理日期和时间格式
    preprocessed_text = preprocess_text(text)
    
    # 设计prompt，包含in-context learning示例
    examples = [
        {"input": "在2023/24学年，他的绩点高达3.9", "output": "在二零二三、二四学年，他的绩点高达三点九"},
        {"input": "2023年GDP增长5.2%", "output": "二零二三年GDP增长百分之五点二"},
        {"input": "今年321-839工程的进展很顺利", "output": "今年三二一杠八三九工程的进展很顺利"},
        {"input": "这个人群的占比常年保持在32-46%", "output": "这个人群的占比常年保持在百分之三十二至百分之四十六"},
        {"input": "1909-1983年，他完成了这项工作", "output": "一九零九至一九八三年，他完成了这项工作"}
    ]
    
    examples_text = "\n".join([f"输入: {e['input']}\n输出: {e['output']}" for e in examples])
    
    prompt = f"""

                以下是一些例子:
                {examples_text}

                现在请转换这个文本:
                {preprocessed_text}

                请将转换后的文本放在三引号内:
                '''
            """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个强大的文字转语音系统，请重复下面的文本，同时把数字内容转换为清楚、自然地普通话读法，保留原文本中的非数字相关的内容不变，将数字用中文读法写出。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            stream=False
        )
        
        response_text = response.choices[0].message.content
        
        # 提取三引号中的内容
        start_marker = "'''"
        end_marker = "'''"
        start_index = response_text.find(start_marker) + len(start_marker)
        end_index = response_text.rfind(end_marker)
        
        if start_index != -1 and end_index != -1:
            extracted_text = response_text[start_index:end_index].strip()
            return extracted_text
        else:
            # 如果没有找到三引号，返回整个响应内容
            return response_text.strip()
    except Exception as e:
        print(f"API调用错误: {e}")
        return None

def process_file(input_file, output_file, start_index=0):
    # 判断是否从头开始处理
    first_write = not os.path.exists(output_file) or start_index == 0
    
    # 如果是新文件，创建并写入JSON数组开始
    if first_write:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
        processed_count = 0
    else:
        # 文件已存在且不从头开始，从文件末尾删除结束标记']'
        # 先读取文件内容
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 如果文件以']'结尾，删除它
        if content.endswith(']'):
            content = content[:-1]
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 假设已处理数量就是start_index
        processed_count = start_index

    # 读取JSONL文件
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 忽略空行
                data.append(json.loads(line))
    
    # 计算起始索引和总数
    total = len(data)
    print(f"从第 {start_index+1} 条数据开始处理，共 {total} 条数据")
    
    # 遍历每条数据，从start_index开始
    for i, item in enumerate(tqdm(data[start_index:], desc="处理数据", initial=start_index, total=total)):
        # 直接使用原始文本而不是预处理后的文本
        original_text = item["text"]
        converted_text = call_deepseek_api(original_text)
        
        # 如果API调用成功，写入结果到文件
        if converted_text:
            result = {
                "original": original_text,
                "normalized": converted_text
            }
            
            # 将结果追加到文件
            with open(output_file, 'a', encoding='utf-8') as f:
                # 添加逗号分隔，除了第一条记录
                if processed_count > 0:
                    f.write(',\n')
                
                # 写入单条结果
                json.dump(result, f, ensure_ascii=False, indent=2)
                
                # 增加处理计数
                processed_count += 1
            
        
        time.sleep(0.05)
    
    # 完成后关闭JSON数组
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    print(f"处理完成，共转换 {processed_count} 条记录，已保存到 {output_file}")

if __name__ == "__main__":
    input_file = "numerical/CN_TN_epoch-01-28645_2.jsonl"  # 修改为原始数据文件
    output_file = "numerical/normalized_text.json"  # 修改输出文件名
    
    # 从第0条数据开始处理
    start_index = 0
    
    # 处理文件
    process_file(input_file, output_file, start_index) 
