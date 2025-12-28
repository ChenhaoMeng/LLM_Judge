# 模型比较分析工具

这是一个用于科学地比较两个模型在特定测试集上表现差异的自动化分析工具。

## 功能特性

- **精确对齐**: 使用 `pair_id` 或原始行号确保两个模型在完全相同的样本上进行对比
- **多维度分析**: 从总体表现、现象级别、UID级别等多个维度分析模型差异
- **统计检验**: 使用 McNemar 检验验证性能提升的显著性
- **可视化报告**: 生成图表和详细分析报告
- **灵活配置**: 支持配置文件和命令行参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法1: 使用配置文件

```bash
python improved_analysis.py --config config.json
```

### 方法2: 使用命令行参数

```bash
python improved_analysis.py \
  --small-model Qwen2.5-0.5B_result_full_text.jsonl \
  --large-model Qwen2.5-1.5B_result_full_text.jsonl \
  --output-dir visualizations
```

### 方法3: 直接运行（使用默认配置）

```bash
python improved_analysis.py
```

## 配置文件说明

配置文件 `config.json` 包含以下参数：

- `file_paths`: 指定小模型和大模型的结果文件路径
- `output`: 指定输出文件和目录
- `analysis`: 分析参数，如最小样本数和显著性水平

## 输出文件

运行后会生成以下文件：

1. `analysis_comparison_result.csv` - 所有样本的详细对比数据
2. `analysis_uid_breakdown.csv` - 各UID的准确率统计表
3. `visualizations/model_comparison_analysis.png` - 可视化图表
4. `visualizations/analysis_report.md` - 详细分析报告

## 分析维度

### 1. 总体表现
- 两个模型的整体准确率
- 准确率提升百分比
- McNemar检验的显著性

### 2. 现象级别分析
- 不同语言现象的准确率对比
- 模型在各现象上的提升/下降情况

### 3. UID级别分析
- 每个UID的详细准确率统计
- 提升和倒退最大的UID
- 最小样本数过滤（默认10个样本）

## 可视化内容

- 现象级别的准确率提升柱状图
- 预测一致性分布甜甜圈图
- UID级别提升分布直方图
- 整体统计摘要

## 主要改进

1. **面向对象设计**: 使用类封装分析逻辑，提高代码可维护性
2. **详细统计**: 增加标准差等统计信息
3. **过滤机制**: 对样本数过少的UID进行过滤
4. **命令行支持**: 支持命令行参数和配置文件
5. **详细报告**: 生成Markdown格式的详细分析报告
6. **错误处理**: 增强错误处理和验证
7. **文档完善**: 提供完整的使用说明