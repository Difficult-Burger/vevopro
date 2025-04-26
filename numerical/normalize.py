import json
import os
import re  # 导入正则表达式模块

# 定义标签类型对应的中文口语表达
LABEL_TO_CHINESE = {
    "HYPHEN_RANGE": "至",
    "COLON_HOUR": "点",
    "COLON_MINUTE": "分",
    "SECOND_CARDINAL": "秒",
    "MINUTE_CARDINAL": "分",
    "DAY_CARDINAL": "日",
    "MONTH_CARDINAL": "月",
    "SLASH_YEAR": "年",
    "SLASH_MONTH": "月",
    "SLASH_PER": "每",
    "SLASH_OR": "或",
    "SLASH_FRACTION": "",  # 按照分数读
    "HYPHEN_RATIO": "比",
    "HYPHEN_MINUS": "负",
    "HYPHEN_SUBZERO": "零下",
    "HYPHEN_EXTENSION": "转",
    "POINT": "点",
    "POWER_OPERATOR": "次方",
    "NUM_TWO_LIANG": "两"
}

# 添加距离单位映射表
UNIT_TO_CHINESE = {
    "km": "千米",
    "m": "米",
    "dm": "分米",
    "cm": "厘米",
    "mm": "毫米",
    "nm": "纳米"
}

# 中文单位列表，用于避免重复处理
CHINESE_UNITS = ["千米", "米", "分米", "厘米", "毫米", "纳米"]

def remove_html_tags(text):
    """删除文本中的HTML标签"""
    # 使用正则表达式去除HTML标签，增加更多复杂标签的匹配
    clean_text = re.sub(r'<[^>]*>', '', text)
    
    # 处理HTML实体，如&nbsp; &lt; &gt;等
    clean_text = re.sub(r'&[a-zA-Z]+;', '', clean_text)
    clean_text = re.sub(r'&\#\d+;', '', clean_text)
    
    # 处理URL编码，如%3A等
    clean_text = re.sub(r'%[0-9A-Fa-f]{2}', '', clean_text)
    
    return clean_text

def adjust_label_positions(text, clean_text, labels):
    """调整标签位置，适应HTML标签删除后的文本"""
    # 创建字符映射表，记录原文本中的字符在清理后文本中的位置
    char_map = []
    clean_index = 0
    
    for i, char in enumerate(text):
        if clean_index < len(clean_text) and char == clean_text[clean_index]:
            char_map.append(clean_index)
            clean_index += 1
        else:
            # 这个字符在清理后的文本中不存在，映射为-1
            char_map.append(-1)
    
    # 调整标签位置
    adjusted_labels = []
    for start, end, label_type in labels:
        # 寻找新的开始位置
        new_start = None
        for i in range(start, min(len(char_map), end)):
            if char_map[i] != -1:
                new_start = char_map[i]
                break
                
        # 寻找新的结束位置
        new_end = None
        for i in range(min(len(char_map)-1, end-1), start-1, -1):
            if char_map[i] != -1:
                new_end = char_map[i] + 1
                break
        
        # 如果找到有效的新位置，添加调整后的标签
        if new_start is not None and new_end is not None:
            adjusted_labels.append((new_start, new_end, label_type))
    
    return adjusted_labels

def remove_hour_leading_zero(text, hour_start, hour_end):
    """移除小时位的前导零，但保留单独的0"""
    hour = text[hour_start:hour_end]
    # 去除前导零，但保留单独的0
    hour_no_leading_zero = hour.lstrip('0') or '0'
    # 返回处理后的小时
    return hour_no_leading_zero

def process_time_format(text):
    """处理时间格式，此函数现在只用于测试，实际处理已移至标签处理阶段"""
    # 匹配时间格式 hh:mm
    time_pattern = re.compile(r'(\d{2}):(\d{2})')
    
    def time_replace(match):
        original_hour = match.group(1)
        hour = original_hour.lstrip('0') or '0'  # 去除小时前导零，但保留单独的'0'
        minute = match.group(2)
        return f"{hour}点{minute}分"
            
    result = time_pattern.sub(time_replace, text)
    return result

def process_distance_units(text):
    """处理距离单位，如40mm -> 40毫米"""
    # 检查文本是否包含可能会被错误处理的模式
    # 避免处理已经含有中文单位的文本
    for unit in CHINESE_UNITS:
        if unit in text:
            # 修正重复单位，比如"米m"
            for eng_unit, chn_unit in UNIT_TO_CHINESE.items():
                if chn_unit in text:
                    # 修复类似"米m"这样的重复单位
                    text = re.sub(f"{chn_unit}{eng_unit}", chn_unit, text)
    
    # 处理简单的数字+单位格式
    # 确保数字直接跟随单位，避免处理已转换为"至"的连字符
    simple_unit_pattern = re.compile(r'(\d+)(' + '|'.join(UNIT_TO_CHINESE.keys()) + r')(?![a-zA-Z])')
    
    def simple_unit_replace(match):
        number = match.group(1)
        unit = match.group(2)
        return f"{number}{UNIT_TO_CHINESE[unit]}"
        
    text = simple_unit_pattern.sub(simple_unit_replace, text)
    
    # 处理范围型数字+单位格式 (如 17-40mm)
    range_unit_pattern = re.compile(r'(\d+)[-至](\d+)(' + '|'.join(UNIT_TO_CHINESE.keys()) + r')(?![a-zA-Z])')
    
    def range_unit_replace(match):
        start_num = match.group(1)
        end_num = match.group(2)
        unit = match.group(3)
        return f"{start_num}至{end_num}{UNIT_TO_CHINESE[unit]}"
        
    text = range_unit_pattern.sub(range_unit_replace, text)
    
    return text

def process_date_format(text):
    """处理日期格式，去除月份和日期的前导零"""
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

def normalize_text(text, labels):
    """根据标签将原始文本转换为口语化表达"""
    # 首先删除HTML标签
    clean_text = remove_html_tags(text)
    
    # 调整标签位置
    adjusted_labels = labels
    if clean_text != text:
        adjusted_labels = adjust_label_positions(text, clean_text, labels)
    
    # 按照位置排序标签
    sorted_labels = sorted(adjusted_labels, key=lambda x: x[0])
    
    # 创建修改列表
    modifications = []
    
    # 识别时间格式的标签，预处理小时位
    time_labels = {}  # 存储时间标签 {hour_position: (hour_start, hour_end, colon_position)}
    
    # 第一轮：查找时间相关标签
    for i, (start, end, label_type) in enumerate(sorted_labels):
        # 确保标签位置不超出文本长度
        if start >= len(clean_text) or end > len(clean_text):
            continue
            
        # 识别时间格式："xx:xx"中的冒号
        if label_type == "COLON_HOUR":
            # 小时位应该在冒号之前2位
            hour_start = max(0, start - 2)
            hour_end = start
            
            # 检查小时位是否为两位数字
            if hour_start < hour_end and re.match(r'^\d{2}$', clean_text[hour_start:hour_end]):
                # 存储小时标签位置和冒号位置
                time_labels[hour_start] = (hour_start, hour_end, start)
    
    # 第二轮：处理所有标签
    for start, end, label_type in sorted_labels:
        # 确保标签位置不超出文本长度
        if start >= len(clean_text) or end > len(clean_text):
            continue
            
        if label_type == "HYPHEN_IGNORE":
            # 将HYPHEN-IGNORE标签处理为"杠"，而非忽略
            modifications.append((start, end, "杠"))
        elif label_type == "PUNC":
            # 只有被特别标记为PUNC的标点符号才处理，正常标点保留
            segment = clean_text[start:end]
            # 如果是需要特殊处理的标点（在LABEL_TO_CHINESE中有定义），则进行替换
            # 否则保留原标点
            if segment in LABEL_TO_CHINESE:
                modifications.append((start, end, LABEL_TO_CHINESE.get(segment, segment)))
        elif label_type in LABEL_TO_CHINESE and LABEL_TO_CHINESE[label_type]:
            # 有特定读法的标签
            if label_type == "HYPHEN_RANGE":
                # 范围标记，替换为"至"
                modifications.append((start, end, LABEL_TO_CHINESE[label_type]))
            elif label_type == "COLON_HOUR":
                # 时间冒号(小时)，这里将小时位的前导零去除
                # 检查该冒号是否是之前识别的时间格式一部分
                for hour_start, (h_start, h_end, colon_pos) in time_labels.items():
                    if colon_pos == start:  # 找到匹配的时间格式
                        # 处理小时位前导零
                        hour_no_leading_zero = remove_hour_leading_zero(clean_text, h_start, h_end)
                        # 替换原小时数字
                        modifications.append((h_start, h_end, hour_no_leading_zero))
                        break
                        
                # 冒号替换为"点"
                modifications.append((start, end, LABEL_TO_CHINESE[label_type]))
            elif label_type == "COLON_MINUTE":
                # 时间冒号(分钟)
                modifications.append((start, end, LABEL_TO_CHINESE[label_type]))
            elif label_type == "POINT":
                # 小数点
                modifications.append((start, end, LABEL_TO_CHINESE[label_type]))
            elif label_type in ["SECOND_CARDINAL", "MINUTE_CARDINAL", "DAY_CARDINAL", "MONTH_CARDINAL"]:
                # 需要添加单位的基数
                segment = clean_text[start:end]
                suffix = LABEL_TO_CHINESE[label_type]
                modifications.append((start, end, segment + suffix))
            elif label_type == "NUM_TWO_LIANG":
                # 数字2特殊读法
                modifications.append((start, end, LABEL_TO_CHINESE[label_type]))
            else:
                # 其他有特定读法的情况
                modifications.append((start, end, LABEL_TO_CHINESE[label_type]))
        elif label_type == "SLASH_FRACTION" and "/" in clean_text[start:end]:
            # 处理分数形式，如 3/4 读作 "四分之三"
            parts = clean_text[start:end].split("/")
            if len(parts) == 2:
                modifications.append((start, end, f"{parts[1]}分之{parts[0]}"))
    
    # 按位置降序排序修改列表，从后向前应用修改
    modifications.sort(key=lambda x: x[0], reverse=True)
    
    # 应用修改
    result = list(clean_text)
    
    # 从后向前应用修改，避免位置偏移问题
    for start, end, replacement in modifications:
        result[start:end] = replacement
    
    # 将列表转换回字符串
    normalized_text = "".join(result)
    
    # 然后处理日期格式
    normalized_text = process_date_format(normalized_text)
    
    # 最后处理距离单位（确保在其他处理完成后再处理单位，避免干扰）
    normalized_text = process_distance_units(normalized_text)
    
    return normalized_text

def process_dataset(input_file, output_file):
    """处理整个数据集并输出结果"""
    normalized_data = []
    count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                try:
                    data = json.loads(line)
                    original_text = data["text"]
                    labels = data["labels"]
                    
                    # 归一化处理
                    normalized_text = normalize_text(original_text, labels)
                    
                    # 添加到结果
                    normalized_data.append({
                        "original": original_text,
                        "normalized": normalized_text
                    })
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"已处理 {count} 条数据")
                except Exception as e:
                    error_count += 1
                    print(f"处理错误 (总计 {error_count} 个): {str(e)[:100]}... 在行: {line[:50]}...")
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共处理 {len(normalized_data)} 条数据，跳过 {error_count} 条错误数据")

if __name__ == "__main__":
    # 添加测试时间格式处理函数
    test_times = ["06:00", "06:03", "10:00", "00:30"]
    print("测试时间格式处理：")
    for time in test_times:
        result = process_time_format(time)
        print(f"测试结果 {time} -> {result}")
    
    input_file = "CN_TN_epoch-01-28645_2.jsonl"
    output_file = "normalized.json"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
    else:
        process_dataset(input_file, output_file) 