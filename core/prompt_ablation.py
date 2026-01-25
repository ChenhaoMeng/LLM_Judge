"""
消融实验：测试不同 Prompt 模板对 PPL 评分的影响

目的：
- 评估模型的鲁棒性
- 测试不同 prompt 格式对评分结果的影响
- 识别对 prompt 敏感的语法现象
"""

import os
os.environ['HF_HOME'] = '/mnt/drive1/chenhao/cache/huggingface'

import torch
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ================= ⚙️ 配置 =================

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

INPUT_FILE = os.path.join(PROJECT_ROOT, "data/ellipsis_adj.jsonl")  # 选择一个代表性的文件
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_report")
MODEL_KEY = "qwen-1.5b"  # 测试模型

MODEL_PATHS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-7b": "Qwen/Qwen2.5-7B",
    "qwen-3b": "Qwen/Qwen2.5-3B",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}

# ================= 🔬 Prompt 模板变体 =================

PROMPT_VARIANTS = {
    "baseline": {
        "name": "基线（无 prompt）",
        "format": lambda sentence: sentence
    },
    "task_explicit": {
        "name": "显式任务描述",
        "format": lambda sentence: f"请判断以下句子的语法正确性：{sentence}"
    },
    "instruction_style": {
        "name": "指令风格",
        "format": lambda sentence: f"任务：语法判断\n句子：{sentence}"
    },
    "qa_style": {
        "name": "问答风格",
        "format": lambda sentence: f"Q: 这个句子语法正确吗？\nA: {sentence}"
    },
    "comparison_style": {
        "name": "对比风格",
        "format": lambda sentence: f"分析以下句子的语法：{sentence}\n请给出你的判断。"
    },
    "formal_style": {
        "name": "正式风格",
        "format": lambda sentence: f"请进行语法分析：{sentence}"
    },
    "casual_style": {
        "name": "随意风格",
        "format": lambda sentence: f"看看这个句子：{sentence}"
    }
}

# ================= 🛠️ 模型评分器 =================

class PromptAblationScorer:
    def __init__(self, model_key, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = MODEL_PATHS.get(model_key, model_key)
        self.device = device
        print(f"🚀 [Loading] 正在加载模型: {self.model_name} ...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def get_sentence_score(self, sentence):
        """计算句子的 perplexity 分数"""
        if not sentence:
            return -float('inf')
        
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.model.device)
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            score = -loss.item()
        
        return score

# ================= 🔬 消融实验 =================

def run_ablation_experiment(input_file, model_key, output_dir):
    """运行消融实验"""
    
    print("="*60)
    print("🔬 Prompt 消融实验")
    print("="*60)
    
    # 加载数据
    print(f"📂 加载数据: {input_file}")
    try:
        df = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return
    
    print(f"   数据量: {len(df)} 条")
    
    # 加载模型
    scorer = PromptAblationScorer(model_key)
    
    # 结果存储
    results = []
    
    print("\n🔬 开始评估不同 Prompt 变体...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理样本"):
        s_good = row.get('sentence_good', '')
        s_bad = row.get('sentence_bad', '')
        phenomenon = row.get('phenomenon', 'unknown')
        pair_id = row.get('pairID', idx)
        
        if not s_good or not s_bad:
            continue
        
        # 对每个 prompt 变体进行评估
        variant_results = {}
        
        for variant_key, variant_config in PROMPT_VARIANTS.items():
            format_func = variant_config["format"]
            
            # 格式化句子
            s_good_formatted = format_func(s_good)
            s_bad_formatted = format_func(s_bad)
            
            # 计算分数
            score_good = scorer.get_sentence_score(s_good_formatted)
            score_bad = scorer.get_sentence_score(s_bad_formatted)
            
            # 判断是否正确
            is_correct = score_good > score_bad
            
            variant_results[variant_key] = {
                "score_good": score_good,
                "score_bad": score_bad,
                "is_correct": is_correct,
                "score_diff": score_good - score_bad
            }
        
        # 保存结果
        result_row = {
            "pair_id": pair_id,
            "phenomenon": phenomenon,
            "sentence_good": s_good,
            "sentence_bad": s_bad,
        }
        
        for variant_key, variant_result in variant_results.items():
            result_row[f"{variant_key}_score_good"] = variant_result["score_good"]
            result_row[f"{variant_key}_score_bad"] = variant_result["score_bad"]
            result_row[f"{variant_key}_is_correct"] = variant_result["is_correct"]
            result_row[f"{variant_key}_score_diff"] = variant_result["score_diff"]
        
        results.append(result_row)
    
    # 转换为 DataFrame
    df_results = pd.DataFrame(results)
    
    # ================= 📊 统计分析 =================
    
    print("\n📊 统计分析...")
    
    # 1. 计算每个变体的准确率
    accuracy_summary = []
    for variant_key in PROMPT_VARIANTS.keys():
        correct_col = f"{variant_key}_is_correct"
        if correct_col in df_results.columns:
            accuracy = df_results[correct_col].mean()
            accuracy_summary.append({
                "Variant": PROMPT_VARIANTS[variant_key]["name"],
                "Variant_Key": variant_key,
                "Accuracy": accuracy,
                "Total": len(df_results),
                "Correct": df_results[correct_col].sum()
            })
    
    df_accuracy = pd.DataFrame(accuracy_summary).sort_values("Accuracy", ascending=False)
    
    # 2. 计算分数差异的统计量
    score_diff_summary = []
    for variant_key in PROMPT_VARIANTS.keys():
        diff_col = f"{variant_key}_score_diff"
        if diff_col in df_results.columns:
            diff_values = df_results[diff_col].values
            score_diff_summary.append({
                "Variant": PROMPT_VARIANTS[variant_key]["name"],
                "Mean_Diff": np.mean(diff_values),
                "Std_Diff": np.std(diff_values),
                "Min_Diff": np.min(diff_values),
                "Max_Diff": np.max(diff_values),
                "Median_Diff": np.median(diff_values)
            })
    
    df_diff = pd.DataFrame(score_diff_summary)
    
    # 3. 计算鲁棒性（不同 prompt 之间的一致性）
    # 使用方差来衡量：方差越小，说明不同 prompt 的结果越一致
    robustness_summary = []
    for idx, row in df_results.iterrows():
        correct_values = []
        for variant_key in PROMPT_VARIANTS.keys():
            correct_col = f"{variant_key}_is_correct"
            if correct_col in df_results.columns:
                correct_values.append(row[correct_col])
        
        if correct_values:
            # 一致性：所有 prompt 都给出相同判断的比例
            consistency = 1.0 if len(set(correct_values)) == 1 else 0.0
            robustness_summary.append({
                "pair_id": row["pair_id"],
                "consistency": consistency,
                "num_variants_agree": sum(correct_values) if all(correct_values) else (len(correct_values) - sum(correct_values))
            })
    
    df_robustness = pd.DataFrame(robustness_summary)
    overall_consistency = df_robustness["consistency"].mean()
    
    # 4. 显著性检验：比较 baseline 与其他变体
    statistical_tests = []
    baseline_col = "baseline_is_correct"
    
    if baseline_col in df_results.columns:
        baseline_scores = df_results[baseline_col].astype(int).values
        
        for variant_key in PROMPT_VARIANTS.keys():
            if variant_key == "baseline":
                continue
            
            variant_col = f"{variant_key}_is_correct"
            if variant_col in df_results.columns:
                variant_scores = df_results[variant_col].astype(int).values
                
                # McNemar 检验（配对样本的非参数检验）
                try:
                    from scipy.stats import mcnemar
                    contingency_table = pd.crosstab(
                        df_results[baseline_col], 
                        df_results[variant_col]
                    ).values
                    if contingency_table.shape == (2, 2):
                        stat, p_value = mcnemar(contingency_table, exact=True)
                    else:
                        p_value = 1.0
                except:
                    p_value = 1.0
                
                # 准确率差异
                acc_diff = df_results[variant_col].mean() - df_results[baseline_col].mean()
                
                statistical_tests.append({
                    "Variant": PROMPT_VARIANTS[variant_key]["name"],
                    "Accuracy_Diff": acc_diff,
                    "P_Value": p_value,
                    "Significant": "Yes" if p_value < 0.05 else "No"
                })
    
    df_stats = pd.DataFrame(statistical_tests)
    
    # ================= 💾 保存结果 =================
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    detailed_path = os.path.join(output_dir, f"prompt_ablation_detailed_{model_key.replace('-', '_')}.csv")
    df_results.to_csv(detailed_path, index=False, encoding='utf-8')
    
    # 保存准确率摘要
    accuracy_path = os.path.join(output_dir, f"prompt_ablation_accuracy_{model_key.replace('-', '_')}.csv")
    df_accuracy.to_csv(accuracy_path, index=False, encoding='utf-8')
    
    # 保存分数差异摘要
    diff_path = os.path.join(output_dir, f"prompt_ablation_diff_{model_key.replace('-', '_')}.csv")
    df_diff.to_csv(diff_path, index=False, encoding='utf-8')
    
    # 保存统计检验结果
    if not df_stats.empty:
        stats_path = os.path.join(output_dir, f"prompt_ablation_statistical_tests_{model_key.replace('-', '_')}.csv")
        df_stats.to_csv(stats_path, index=False, encoding='utf-8')
    
    # ================= 📈 可视化 =================
    
    plt.figure(figsize=(14, 8))
    
    # 1. 准确率对比
    plt.subplot(2, 2, 1)
    sns.barplot(data=df_accuracy, x="Variant", y="Accuracy", palette="viridis")
    plt.title("不同 Prompt 变体的准确率对比", fontsize=12, fontweight='bold')
    plt.xlabel("Prompt 变体", fontsize=10)
    plt.ylabel("准确率", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    
    # 2. 分数差异分布
    plt.subplot(2, 2, 2)
    score_diffs = []
    variant_labels = []
    for variant_key in PROMPT_VARIANTS.keys():
        diff_col = f"{variant_key}_score_diff"
        if diff_col in df_results.columns:
            score_diffs.extend(df_results[diff_col].values)
            variant_labels.extend([PROMPT_VARIANTS[variant_key]["name"]] * len(df_results))
    
    df_plot = pd.DataFrame({"Score_Diff": score_diffs, "Variant": variant_labels})
    sns.boxplot(data=df_plot, x="Variant", y="Score_Diff", palette="viridis")
    plt.title("不同 Prompt 变体的分数差异分布", fontsize=12, fontweight='bold')
    plt.xlabel("Prompt 变体", fontsize=10)
    plt.ylabel("分数差异 (Good - Bad)", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. 一致性热力图
    plt.subplot(2, 2, 3)
    consistency_matrix = []
    variant_names = [PROMPT_VARIANTS[k]["name"] for k in PROMPT_VARIANTS.keys()]
    
    for i, key1 in enumerate(PROMPT_VARIANTS.keys()):
        row = []
        for j, key2 in enumerate(PROMPT_VARIANTS.keys()):
            col1 = f"{key1}_is_correct"
            col2 = f"{key2}_is_correct"
            if col1 in df_results.columns and col2 in df_results.columns:
                agreement = (df_results[col1] == df_results[col2]).mean()
                row.append(agreement)
            else:
                row.append(0.0)
        consistency_matrix.append(row)
    
    sns.heatmap(consistency_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=variant_names, yticklabels=variant_names,
                cbar_kws={'label': '一致性'})
    plt.title("Prompt 变体间判断一致性", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 4. 鲁棒性分析
    plt.subplot(2, 2, 4)
    if not df_robustness.empty:
        consistency_counts = df_robustness["consistency"].value_counts()
        labels = ["不一致" if not c else "一致" for c in consistency_counts.index]
        plt.pie(consistency_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f"整体一致性: {overall_consistency:.2%}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"prompt_ablation_analysis_{model_key.replace('-', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ================= 📋 打印摘要 =================
    
    print("\n" + "="*60)
    print("📊 消融实验结果摘要")
    print("="*60)
    
    print("\n1. 准确率排名:")
    print(df_accuracy[["Variant", "Accuracy", "Correct", "Total"]].to_string(index=False))
    
    print(f"\n2. 整体一致性: {overall_consistency:.2%}")
    print(f"   ({df_robustness['consistency'].sum()}/{len(df_robustness)} 个样本在所有 prompt 下结果一致)")
    
    if not df_stats.empty:
        print("\n3. 与 Baseline 的统计显著性检验:")
        print(df_stats.to_string(index=False))
    
    print(f"\n💾 详细结果已保存到: {output_dir}")
    print(f"   - 详细数据: {detailed_path}")
    print(f"   - 准确率摘要: {accuracy_path}")
    print(f"   - 可视化图表: {plot_path}")
    
    # 清理
    del scorer
    torch.cuda.empty_cache()

# ================= 🏁 主函数 =================

if __name__ == "__main__":
    run_ablation_experiment(INPUT_FILE, MODEL_KEY, OUTPUT_DIR)
