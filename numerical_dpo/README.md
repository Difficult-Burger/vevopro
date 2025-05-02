# 中文数字朗读DPO训练

本项目使用Direct Preference Optimization (DPO)技术增强Ints TTS模型朗读中文数字的准确性。

## 项目结构

```
numerical_dpo/
├── README.md                   # 本文件
├── requirements.txt            # 依赖包列表
├── bins/                       # 可执行程序
│   ├── extract_speech_tokens.py  # 从音频提取语音tokens
│   ├── test_dpo_implementation.py # 测试DPO实现
│   └── train_dpo.py            # DPO训练主程序
├── configs/                    # 配置文件
│   └── dpo_numerical.json      # DPO训练配置
├── data/                       # 数据目录
│   ├── dataset.py              # 数据集实现
│   ├── dpo_data.json           # 处理后的训练数据
│   ├── test_data.json          # 划分出的测试数据
│   └── prompt_speech/          # 提示语音
├── models/                     # 模型实现
│   └── dpo_model.py            # DPO模型类
├── scripts/                    # 脚本
│   ├── evaluate.py             # 评估脚本
│   ├── prepare_dpo_data.py     # 数据准备脚本
│   ├── run_numerical_dpo.sh    # 主运行脚本
│   └── test_pipeline.sh        # 测试脚本
├── utils/                      # 工具函数
│   ├── data_processor.py       # 数据处理工具
│   └── dpo_utils.py            # DPO工具函数
└── logs/                       # 日志目录
```

## 安装

```bash
# 克隆项目
git clone <repository_url>
cd <repository_directory>

# 安装依赖
pip install -r numerical_dpo/requirements.txt
```

## 使用方法

### 1. 准备数据

将normalized_text.json放在numerical目录下，文件格式为：

```json
[
  {
    "original": "原始文本，包含数字",
    "normalized": "标准化文本，数字已规范化"
  },
  ...
]
```

### 2. 运行DPO训练

```bash
bash numerical_dpo/scripts/run_numerical_dpo.sh
```

### 3. 测试流程

```bash
bash numerical_dpo/scripts/test_pipeline.sh
```

## 数据集划分

系统会自动将数据集划分为：
- **训练集(80%)**: 用于DPO训练
- **测试集(20%)**: 用于模型评估

划分比例可通过参数`--test_split`调整（默认0.2），测试集保存为`data/test_data.json`。

## DPO训练原理

1. **数据准备**：使用normalized文本生成的语音作为正样本(chosen)，original文本生成的语音作为负样本(rejected)
2. **模型训练**：使用DPO算法，让模型学习朗读数字的正确方式
3. **评估**：使用划分出的测试集对比训练前后的模型，验证数字朗读质量的提升

## 关键参数

- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--device`: 使用的设备 (如"cuda:0"或"cpu")
- `--mock_data`: 使用模拟数据进行测试
- `--test_split`: 测试集划分比例（默认0.2，表示20%）
- `--seed`: 随机种子，确保数据划分可复现（默认42）

## 注意事项

1. 确保模型路径正确，默认使用"Ints/ins_model_dpo_0228"
2. 训练需要GPU资源，如果没有GPU，可以将device设置为"cpu"，但会很慢
3. 训练结果保存在numerical_dpo/runs/numerical/目录下
4. 评估结果使用划分好的测试集，确保模型在未见过的数据上表现良好 