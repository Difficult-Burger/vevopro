#!/bin/bash

# 测试DPO训练流程的脚本
set -e

# 使用模拟数据进行快速测试
echo "=== 测试DPO训练流程 ==="

# 测试目录
TEST_DIR="numerical_dpo/tests"
mkdir -p "$TEST_DIR"

# 创建测试提示
echo "这是测试提示" > "$TEST_DIR/prompt.txt"

# 1. 测试数据准备
echo "测试数据准备..."
python numerical_dpo/scripts/prepare_dpo_data.py \
    --ints_model_path "Ints/ins_model_dpo_0228" \
    --w2v_bert_path "Ints/w2v-bert-2" \
    --dual_codec_path "Ints/SpeechGeneration-dev-ins/RepCodec" \
    --tokenizer_path "Ints/ins_model_dpo_0228" \
    --normalized_text_path "numerical/normalized_text.json" \
    --prompt_wav_path "numerical_dpo/data/prompt_speech/prompt.wav" \
    --prompt_text_path "$TEST_DIR/prompt.txt" \
    --output_dir "$TEST_DIR/dpo_data" \
    --output_json "$TEST_DIR/dpo_data.json" \
    --max_samples 5 \
    --device "cpu" \
    --mock_data

# 确认生成的文件
if [ -f "$TEST_DIR/dpo_data.json" ]; then
    echo "✓ 数据准备测试通过"
else
    echo "✗ 数据准备测试失败"
    exit 1
fi

# 2. 测试数据加载
echo "测试数据加载..."
python -c "
import sys
sys.path.append('.')
from numerical_dpo.data.dataset import create_dpo_dataloader
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained('Ints/ins_model_dpo_0228')

# 创建dataloader
dataloader = create_dpo_dataloader(
    json_path='$TEST_DIR/dpo_data.json',
    tokenizer=tokenizer,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# 检查dataloader
print(f'数据集大小: {len(dataloader.dataset)}')
batch = next(iter(dataloader))
print(f'批次中的字段: {list(batch.keys())}')
print('✓ 数据加载测试通过')
"

if [ $? -ne 0 ]; then
    echo "✗ 数据加载测试失败"
    exit 1
fi

# 3. 测试模型加载（如果模型存在）
echo "测试模型加载..."
python -c "
import sys
sys.path.append('.')
from numerical_dpo.models.dpo_model import DPOIntsModel
import os

# 检查模型路径是否存在
model_path = 'Ints/ins_model_dpo_0228'
if os.path.exists(model_path):
    try:
        # 尝试加载模型
        model = DPOIntsModel(model_path=model_path, device='cpu')
        print('✓ 模型加载测试通过')
    except Exception as e:
        print(f'✗ 模型加载测试失败: {e}')
else:
    print(f'模型路径 {model_path} 不存在，跳过模型加载测试')
"

echo "=== 测试完成 ===" 