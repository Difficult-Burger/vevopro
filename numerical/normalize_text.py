import json
import time
import os
from tqdm import tqdm
from openai import OpenAI
# from config import BASE_URL, API_KEY

# 初始化OpenAI客户端 - 新的调用方式
client = OpenAI(
    api_key="sk-aakjkyvxyuthakahaftslxmryoxntklnxvkgfzalyicfvhkm",
    base_url="https://api.siliconflow.cn/v1"
)

# 原来的客户端初始化方式
# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_deepseek_api(text):
    """调用DeepSeek API将数字转为汉字表达"""
    # 设计prompt，包含in-context learning示例
    examples = [
        {"input": "我有2个苹果和3个橙子", "output": "我有两个苹果和三个橙子"},
        {"input": "这款手机售价1299元", "output": "这款手机售价一千二百九十九元"},
        {"input": "2023年GDP增长5.2%", "output": "二零二三年GDP增长百分之五点二"}
    ]
    
    examples_text = "\n".join([f"输入: {e['input']}\n输出: {e['output']}" for e in examples])
    
    prompt = f"""请将以下文本中的数字转换成汉字表达，保持普通话口语的自然表达方式。注意保留所有非数字内容不变。

以下是一些例子:
{examples_text}

现在请转换这个文本:
{text}

请将转换后的文本放在三引号内:
'''"""

    try:
        # 新的API调用方式
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "你是一个将数字转换为汉字表达的助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # 原来的API调用方式
        # response = client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[
        #         {"role": "system", "content": "你是一个将数字转换为汉字表达的助手"},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.1,
        #     stream=False
        # )
        
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
    # 读取输入JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
    
    # 计算起始索引和总数
    total = len(data)
    print(f"从第 {start_index+1} 条数据开始处理，共 {total} 条数据")
    
    # 遍历每条数据，从start_index开始
    for i, item in enumerate(tqdm(data[start_index:], desc="处理数据", initial=start_index, total=total)):
        normalized_text = item["normalized"]
        converted_text = call_deepseek_api(normalized_text)
        
        # 如果API调用成功，写入结果到文件
        if converted_text:
            result = {
                "original": item["normalized"],
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
        
        # 添加短暂延迟，避免API限制
        time.sleep(0.06)
    
    # 完成后关闭JSON数组
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    print(f"处理完成，共转换 {processed_count} 条记录，已保存到 {output_file}")

if __name__ == "__main__":
    input_file = "numerical/normalized.json"
    output_file = "numerical/normalized_no_num.json"
    
    # 从第238条数据开始继续处理（索引从0开始，所以是237）
    start_index = 450
    
    # 处理文件
    process_file(input_file, output_file, start_index) 