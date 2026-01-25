# LLM Judge: 中文语法能力评估系统

> 一个系统化的研究项目，用于评估大语言模型对中文语法现象的判断能力

## 📋 项目概述

本项目通过 perplexity 评分、Scaling Law 分析、LoRA 微调和 Token Fertility 研究，系统性地评估不同规模的中文语言模型在处理代词指代、省略、量词逻辑等语法现象时的表现。

## 🏗️ 项目结构

```
LLM_Judge/
├── README.md                    # 本文档
├── requirements.txt             # 依赖项
├── config.json                  # 配置文件
│
├── data/                        # 原始数据
│   ├── anaphor_*.jsonl         # 代词指代
│   ├── ellipsis_*.jsonl        # 省略现象
│   └── superlative_quantifiers_*.jsonl  # 量词逻辑
│
├── scripts/                     # 辅助脚本
│   ├── merge.py                # 结果合并
│   ├── fertility.py            # Token生育率分析
│   └── lora_stat.py            # 测试结果统计
│
├── core/                        # 核心功能
│   ├── judge.py                # 模型评估（SLLN-LP归一化）
│   ├── analyser.py             # 统计分析（Bootstrap、Kappa）
│   ├── finetune_dataset.py     # 数据集构建（EtP支持）
│   ├── lora_train.py           # LoRA批量训练
│   ├── poco_train.py           # PoCO两阶段训练
│   ├── lora_test.py            # 微调后测试
│   └── prompt_ablation.py      # Prompt消融实验
│
├── logs/                        # 训练日志
│   ├── scaling_law_training_log.csv
│   └── poco_training_log.csv
│
├── results_scoring/             # 原始评估结果
├── results_merged/              # 合并后的结果
├── analysis_report/             # 分析报告和可视化
└── lora_test_results/           # 微调测试结果
```

## 🚀 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 核心功能

#### 模型评估（带SLLN-LP长度归一化）
```bash
python core/judge.py
```
- 使用 SLLN-LP 处理省略等长度变化现象
- 支持 Qwen、Pythia 系列模型

#### 结果分析与统计
```bash
python core/analyser.py
```
- Bootstrap 置信区间
- 统计显著性检验（Mann-Whitney U）
- 一致性度量（Cohen's Kappa）

#### 数据集构建
```bash
python core/finetune_dataset.py
```
- 支持 Explain-then-Process (EtP) 模式
- 生成多任务混合数据集

#### LoRA 微调训练
```bash
# 标准训练
python core/lora_train.py

# PoCO两阶段训练（推荐）
python core/poco_train.py
```

#### Prompt 消融实验
```bash
python core/prompt_ablation.py
```

## 📊 核心特性

### 1. SLLN-LP 长度归一化
针对省略现象导致的句子长度差异，使用长度归一化公式提升判断公平性：
```python
normalized_score = raw_score / (length ^ alpha)
```

### 2. Explain-then-Process (EtP)
要求模型先解释错误原因，再输出修正结果，提升改错能力：
```
错误原因：{explanation}
修正后的句子：{corrected}
```

### 3. PoCO 两阶段训练
- **Stage 1**: 重点训练改错能力（改错题权重50%）
- **Stage 2**: 平衡所有任务（恢复原始权重）
- 预期效果：改错题准确率提升15-25%

### 4. 统计分析增强
- **Bootstrap置信区间**: 95%置信区间评估
- **统计显著性**: Mann-Whitney U检验
- **一致性度量**: Cohen's Kappa, Scott's Pi

## 📈 主要结果

### Scaling Law 发现
- **Qwen系列**: 整体表现优于Pythia（64.43% vs 43.90%，14B规模）
- **不同现象难度**: 代词指代（74-80%）> 省略（50-54%）> 量词（40-66%）

### 微调效果
- **选择题**: 100%准确率（所有模型）
- **改错题**: 0.5B模型0%，1.5B模型60.85%（PoCO训练后预期提升）

## 🔬 研究问题与改进

### 当前研究重点
1. 中文模型的语法判断能力评估
2. Scaling Law 在不同语法现象上的表现
3. Tokenization 对性能的影响（Token Fertility）
4. 微调策略对任务平衡的影响

### 改进方向
- [x] SLLN-LP 长度归一化
- [x] Explain-then-Process (EtP)
- [x] PoCO 两阶段训练
- [x] 统计显著性检验
- [x] Prompt 鲁棒性分析

详细改进说明见下方章节。

## 📚 详细文档

### 算法升级
见 [UPGRADE_NOTES.md](docs/UPGRADE_NOTES.md) - SLLN-LP、EtP、路径修复

### 高级功能
见 [ADVANCED_IMPROVEMENTS.md](docs/ADVANCED_IMPROVEMENTS.md) - PoCO架构、统计分析、消融实验

### 研究改进建议
见 [RESEARCH_IMPROVEMENTS.md](docs/RESEARCH_IMPROVEMENTS.md) - 深度分析和改进方向

## 🛠️ 配置说明

### 评估配置 (judge.py)
```python
use_length_normalization=True   # 启用SLLN-LP
length_penalty_alpha=0.6        # 长度惩罚系数
```

### 微调配置 (finetune_dataset.py)
```python
USE_ETP = True                  # 启用Explain-then-Process
TASK_WEIGHTS = [0.4, 0.3, 0.3] # 选择题:判断题:改错题
```

### PoCO配置 (poco_train.py)
```python
POCO_CONFIG = {
    "stage1_weight_correction": 0.5,  # Stage1改错题权重
    "stage1_epochs": 2,
    "stage2_epochs": 1,
}
```

## 📊 输出文件说明

### 评估结果
- `results_scoring/{model}/*.jsonl`: 原始评分结果
- `results_merged/{model}_merged.jsonl`: 合并结果

### 分析报告
- `analysis_report/final_accuracy_report.csv`: 准确率报告
- `analysis_report/scaling_law_*.png`: Scaling Law图表
- `analysis_report/statistical_significance_test.csv`: 统计检验
- `analysis_report/agreement_metrics.csv`: 一致性度量

### 训练日志
- `logs/scaling_law_training_log.csv`: 标准训练日志
- `logs/poco_training_log.csv`: PoCO训练日志

## 🔧 常见问题

### Q: 如何选择最佳的Prompt格式？
A: 运行 `prompt_ablation.py` 查看不同Prompt的准确率和一致性。

### Q: PoCO训练比标准训练慢多少？
A: 约增加50%时间（两阶段训练），但改错题准确率提升明显。

### Q: 如何解读统计显著性检验？
A: P < 0.05 表示显著差异，Cohen's d > 0.5 表示中等以上效应量。

## 📝 引用

如果使用本项目，请引用：

```bibtex
@software{llm_judge,
  title = {LLM Judge: Chinese Grammar Evaluation System},
  year = {2025},
  author = {Your Name}
}
```

## 📄 许可证

[您的许可证]

---

**最后更新**: 2025-01-XX
