#!/bin/bash

# DPO训练脚本，用于中文数字朗读
set -e

# 基础配置
INTS_MODEL_PATH="Ints/ins_model_dpo_0228"  # Ints模型路径
W2V_BERT_PATH="Ints/w2v-bert-2"            # wav2vec-BERT模型路径
DUAL_CODEC_PATH="Ints/SpeechGeneration-dev-ins/RepCodec"  # RepCodec路径
TOKENIZER_PATH="$INTS_MODEL_PATH"          # 分词器路径，默认与模型路径相同
NORMALIZED_TEXT_PATH="numerical/normalized_text.json"  # 标准化文本路径

# 数据和输出目录
DATA_DIR="numerical_dpo/data"              # 数据目录
OUTPUT_DIR="numerical_dpo/runs/numerical"  # 输出目录
LOG_DIR="numerical_dpo/logs"               # 日志目录
CONFIG_PATH="numerical_dpo/configs/dpo_numerical.json"  # 配置文件路径

# 提示音频和文本
PROMPT_WAV="numerical_dpo/data/prompt_speech/prompt.wav"  # 提示音频
PROMPT_TEXT="numerical_dpo/data/prompt_speech/prompt.txt"  # 提示文本

# 训练配置
BATCH_SIZE=2                              # 批次大小
NUM_EPOCHS=3                              # 训练轮数
LEARNING_RATE=5e-7                        # 学习率
GRADIENT_ACCUMULATION=4                   # 梯度累积步数
DEVICE="cuda:0"                           # 设备，如果没有GPU，可以设置为"cpu"
MOCK_DATA=false                           # 是否使用模拟数据
TEST_SPLIT=0.2                            # 测试集比例
RANDOM_SEED=42                            # 随机种子

# 创建必要的目录
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$DATA_DIR/prompt_speech"

# 检查提示音频和文本
if [ ! -f "$PROMPT_WAV" ]; then
    echo "提示音频文件不存在，创建默认提示..."
    # 如果没有提示音频，创建一个默认的提示文本
    echo "请朗读以下文本" > "$PROMPT_TEXT"
    MOCK_DATA=true
fi

# 准备DPO数据
echo "准备DPO训练数据..."
MOCK_DATA_OPTION=""
if [ "$MOCK_DATA" = true ]; then
    MOCK_DATA_OPTION="--mock_data"
fi

python numerical_dpo/scripts/prepare_dpo_data.py \
    --ints_model_path "$INTS_MODEL_PATH" \
    --w2v_bert_path "$W2V_BERT_PATH" \
    --dual_codec_path "$DUAL_CODEC_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --normalized_text_path "$NORMALIZED_TEXT_PATH" \
    --prompt_wav_path "$PROMPT_WAV" \
    --prompt_text_path "$PROMPT_TEXT" \
    --output_dir "$DATA_DIR/dpo_data" \
    --output_json "$DATA_DIR/dpo_data.json" \
    --device "$DEVICE" \
    --test_split "$TEST_SPLIT" \
    --seed "$RANDOM_SEED" \
    $MOCK_DATA_OPTION

# 训练DPO模型
echo "开始DPO训练..."
python numerical_dpo/bins/train_dpo.py \
    --config_path "$CONFIG_PATH" \
    --data_path "$DATA_DIR/dpo_data.json" \
    --model_path "$INTS_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --device "$DEVICE"

# 评估模型
echo "评估DPO模型..."
TEST_DATA_PATH="$DATA_DIR/test_data.json"
python numerical_dpo/scripts/evaluate.py \
    --model_path "$OUTPUT_DIR/final_model" \
    --tokenizer_path "$OUTPUT_DIR/final_model" \
    --test_data "$TEST_DATA_PATH" \
    --prompt_wav_path "$PROMPT_WAV" \
    --prompt_text_path "$PROMPT_TEXT" \
    --output_dir "$OUTPUT_DIR/evaluation" \
    --device "$DEVICE"

echo "DPO训练和评估完成！" 