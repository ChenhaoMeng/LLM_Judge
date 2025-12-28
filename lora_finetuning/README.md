# LoRA 微调实验：语法能力提升研究

## 项目概述

本项目旨在通过 LoRA (Low-Rank Adaptation) 微调技术，研究预训练语言模型在语法任务（特别是 Ellipsis 和 Anaphor）上的表现是否可以通过微调得到改善。

研究问题：模型是"学不会"这些语法结构，还是只是"没见过"足够的相关训练数据？

## 文件结构

```
lora_finetuning/
├── build_dataset.py      # 构建针对 Ellipsis 和 Anaphor 的微调数据集
├── lora_finetuning.py    # LoRA 微调主脚本
├── evaluate_model.py     # 评估模型性能的脚本
├── requirements.txt      # 依赖包列表
└── README.md            # 本说明文件
```

## 实验流程

### 1. 数据集构建

运行 `build_dataset.py` 创建专门针对语法任务的训练数据：

```bash
python build_dataset.py
```

### 2. LoRA 微调

运行 `lora_finetuning.py` 对 Qwen2.5-0.5B 模型进行微调：

```bash
python lora_finetuning.py
```

### 3. 性能评估

运行 `evaluate_model.py` 比较微调前后的模型性能：

```bash
python evaluate_model.py
```

## 数据集说明

数据集包含两类语法现象：

1. **Ellipsis (省略结构)**: 如 "John said Mary would come, and she did come."
2. **Anaphor (回指结构)**: 如 "The cat saw itself in the mirror."

每类包含正确和错误的示例，用于训练模型判断语法正确性。

## 模型配置

- 基础模型：Qwen2.5-0.5B (可选 Qwen2.5-1.5B)
- LoRA 配置：r=8, alpha=32, dropout=0.1
- 训练轮数：3 epochs
- 批处理大小：2 (根据显存调整)

## 预期结果

通过对比微调前后的性能，我们可以判断：
- 如果微调后准确率显著提升，说明模型具备语法能力，只是预训练数据分布导致对齐偏差
- 如果微调后准确率提升有限，说明模型在架构或预训练阶段存在根本性限制