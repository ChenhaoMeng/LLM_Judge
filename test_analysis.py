import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# 配置参数
config = {
    "file_paths": {
        "small_model": "Qwen2.5-0.5B_result_full_text.jsonl",
        "large_model": "Qwen2.5-1.5B_result_full_text.jsonl"
    },
    "output": {
        "comparison_csv": "analysis_comparison_result.csv",
        "uid_breakdown_csv": "analysis_uid_breakdown.csv",
        "visualization_dir": "visualizations"
    },
    "analysis": {
        "min_samples_per_uid": 10,
        "significant_p_value": 0.05
    }
}

def load_results_optimized(filepath):
    print(f"\n📖 [Loading] 处理文件: {filepath}")
    if not os.path.exists(filepath):
        print(f"❌ 错误: 找不到文件 {filepath}")
        return None
    
    try:
        df = pd.read_json(filepath, lines=True)
        if 'is_correct' in df.columns and 'correct' not in df.columns:
            df['correct'] = df['is_correct']
        if 'file_source' not in df.columns:
            df['file_source'] = os.path.basename(filepath)
        df['original_row_index'] = df.index
        return df
    except Exception as e:
        print(f"❌ 读取发生错误: {e}")
        return None

def calculate_mcnemar(matrix_df):
    try:
        b = matrix_df.loc[matrix_df['Category'].str.contains('Fixed'), 'Count'].values[0]
        c = matrix_df.loc[matrix_df['Category'].str.contains('Regression'), 'Count'].values[0]
    except IndexError:
        return 0.0, 1.0
    
    if b + c == 0: 
        return 0.0, 1.0
    
    statistic = (b - c)**2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, 1)
    return statistic, p_value

def analyze_comparison(df_small, df_large, name_s="0.5B", name_l="1.5B"):
    print("="*50)
    print("📊 开始对比分析")
    print("="*50)

    # 1. 数据对齐
    align_keys = ['file_source', 'phenomenon', 'UID']
    if 'pair_id' in df_small.columns and 'pair_id' in df_large.columns:
        align_keys.append('pair_id')
        print("✅ 使用 pair_id 进行精确对齐")
    else:
        align_keys.append('original_row_index')
        print("⚠️ 未发现 pair_id，使用原始行号 (row_index) 对齐")

    meta_cols = align_keys + ['sentence_good', 'sentence_bad'] 
    existing_meta = [c for c in meta_cols if c in df_small.columns]
    
    df_compare = pd.merge(
        df_small[existing_meta + ['correct']],
        df_large[existing_meta + ['correct']],
        on=existing_meta,
        suffixes=(f'_{name_s}', f'_{name_l}')
    )
    print(f"🔗 成功对齐样本数: {len(df_compare)}")

    # 2. 核心计算
    col_s = f'correct_{name_s}'
    col_l = f'correct_{name_l}'
    
    df_compare['both_correct'] = df_compare[col_s] & df_compare[col_l]
    df_compare['both_wrong'] = (~df_compare[col_s]) & (~df_compare[col_l])
    df_compare['fixed'] = (~df_compare[col_s]) & df_compare[col_l]
    df_compare['regression'] = df_compare[col_s] & (~df_compare[col_l])

    # 3. 总体统计
    acc_s = df_compare[col_s].mean()
    acc_l = df_compare[col_l].mean()
    print(f"\n🏆 [总体表现]")
    print(f"  • {name_s} Acc: {acc_s:.2%}")
    print(f"  • {name_l} Acc: {acc_l:.2%}")
    print(f"  • Delta:      {acc_l - acc_s:+.2%}")
    print(f"  • 提升:       {((acc_l - acc_s) / acc_s * 100):+.2f}%")

    # 4. McNemar 检验
    matrix_data = [
        {'Category': 'Both Correct (Easy)', 'Count': df_compare['both_correct'].sum()},
        {'Category': 'Both Wrong (Hard)',   'Count': df_compare['both_wrong'].sum()},
        {'Category': f'{name_l} Fixed (Improved)',     'Count': df_compare['fixed'].sum()},
        {'Category': f'{name_s} Won (Regression)',     'Count': df_compare['regression'].sum()}
    ]
    df_matrix = pd.DataFrame(matrix_data)
    stat, p_val = calculate_mcnemar(df_matrix)
    print(f"\n🧩 [显著性分析] McNemar p-value: {p_val:.4e}")
    print(f"  • 显著性: {'✅ 显著' if p_val < config['analysis']['significant_p_value'] else '❌ 不显著'}")

    # 5. Phenomenon 统计
    phenom_group = df_compare.groupby('phenomenon')[[col_s, col_l]].mean()
    phenom_group['Delta'] = phenom_group[col_l] - phenom_group[col_s]
    phenom_group = phenom_group.sort_values('Delta', ascending=False)

    return df_compare, phenom_group, df_matrix

def analyze_uid_stats(df_compare, name_s="0.5B", name_l="1.5B"):
    """
    计算每个 UID 的详细准确率和 Delta
    """
    print(f"\n🔬 [UID 粒度详细分析]")
    col_s = f'correct_{name_s}'
    col_l = f'correct_{name_l}'

    # 按 Phenomenon 和 UID 分组聚合
    uid_stats = df_compare.groupby(['phenomenon', 'UID']).agg(
        count=('file_source', 'count'),       # 样本数量
        acc_small=(col_s, 'mean'),            # 小模型准确率
        acc_large=(col_l, 'mean'),            # 大模型准确率
        std_small=(col_s, 'std'),             # 小模型标准差
        std_large=(col_l, 'std')              # 大模型标准差
    ).reset_index()

    # 处理标准差为 NaN 的情况
    uid_stats['std_small'] = uid_stats['std_small'].fillna(0)
    uid_stats['std_large'] = uid_stats['std_large'].fillna(0)

    # 计算差值
    uid_stats['delta'] = uid_stats['acc_large'] - uid_stats['acc_small']
    uid_stats['abs_delta'] = abs(uid_stats['delta'])

    # 过滤样本数过少的UID
    min_samples = config['analysis']['min_samples_per_uid']
    uid_stats = uid_stats[uid_stats['count'] >= min_samples]
    print(f"📊 过滤后UID数量: {len(uid_stats)} (最小样本数: {min_samples})")

    # 按提升幅度降序排列
    uid_stats = uid_stats.sort_values('delta', ascending=False)

    # 打印 Top 5 提升
    print(f"\n🚀 提升最大的 Top 5 UID (最小样本数: {min_samples}):")
    top_improvements = uid_stats.head(5)
    if len(top_improvements) > 0:
        print(top_improvements.to_string(index=False, formatters={
            'acc_small': '{:.1%}'.format, 
            'acc_large': '{:.1%}'.format, 
            'delta': '{:+.1%}'.format,
            'std_small': '{:.3f}'.format,
            'std_large': '{:.3f}'.format
        }))
    else:
        print("  没有符合条件的UID")

    # 打印 Top 5 倒退
    print(f"\n📉 倒退最大的 Top 5 UID (最小样本数: {min_samples}):")
    top_regressions = uid_stats.tail(5)
    if len(top_regressions) > 0:
        print(top_regressions.to_string(index=False, formatters={
            'acc_small': '{:.1%}'.format, 
            'acc_large': '{:.1%}'.format, 
            'delta': '{:+.1%}'.format,
            'std_small': '{:.3f}'.format,
            'std_large': '{:.3f}'.format
        }))
    else:
        print("  没有符合条件的UID")

    return uid_stats

def analyze_detailed_statistics(df_compare, name_s="0.5B", name_l="1.5B"):
    """
    计算详细统计信息
    """
    col_s = f'correct_{name_s}'
    col_l = f'correct_{name_l}'
    
    stats = {
        'total_samples': len(df_compare),
        'both_correct': df_compare['both_correct'].sum(),
        'both_wrong': df_compare['both_wrong'].sum(),
        'improved': df_compare['fixed'].sum(),
        'regressed': df_compare['regression'].sum(),
        'small_only_correct': df_compare[col_s].sum() - df_compare['both_correct'].sum(),
        'large_only_correct': df_compare[col_l].sum() - df_compare['both_correct'].sum(),
    }
    
    # 计算比例
    total = stats['total_samples']
    stats['improvement_rate'] = stats['improved'] / total
    stats['regression_rate'] = stats['regressed'] / total
    stats['agreement_rate'] = (stats['both_correct'] + stats['both_wrong']) / total
    
    return stats

def generate_detailed_report(df_compare, df_phenom, df_matrix, df_uid, stats, name_s="0.5B", name_l="1.5B"):
    """
    生成详细分析报告
    """
    os.makedirs(config['output']['visualization_dir'], exist_ok=True)
    report_path = f"{config['output']['visualization_dir']}/analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 模型比较分析报告\n\n")
        f.write("## 1. 概述\n\n")
        f.write(f"本次分析比较了 {name_s} 模型和 {name_l} 模型在测试集上的表现差异。\n\n")
        
        f.write("## 2. 总体表现\n\n")
        f.write(f"- 小模型准确率: {df_phenom.iloc[:, 0].mean():.2%}\n")
        f.write(f"- 大模型准确率: {df_phenom.iloc[:, 1].mean():.2%}\n")
        f.write(f"- 准确率提升: {df_phenom.iloc[:, 1].mean() - df_phenom.iloc[:, 0].mean():+.2%}\n")
        f.write(f"- 总样本数: {stats['total_samples']:,}\n\n")
        
        f.write("## 3. 统计检验结果\n\n")
        stat, p_val = calculate_mcnemar(df_matrix)
        f.write(f"- McNemar检验 p-value: {p_val:.4e}\n")
        f.write(f"- 显著性: {'显著' if p_val < config['analysis']['significant_p_value'] else '不显著'}\n\n")
        
        f.write("## 4. 现象级别分析\n\n")
        f.write("| 现象 | 小模型准确率 | 大模型准确率 | 提升 |\n")
        f.write("|------|-------------|-------------|------|\n")
        for idx, row in df_phenom.head(10).iterrows():
            f.write(f"| {idx} | {row.iloc[0]:.2%} | {row.iloc[1]:.2%} | {row['Delta']:+.2%} |\n")
        f.write("\n")
        
        f.write("## 5. UID级别分析\n\n")
        f.write("### 提升最大的UIDs:\n")
        top_improvements = df_uid.head(5)
        if len(top_improvements) > 0:
            f.write("| 现象 | UID | 样本数 | 小模型准确率 | 大模型准确率 | 提升 |\n")
            f.write("|------|-----|--------|-------------|-------------|------|\n")
            for _, row in top_improvements.iterrows():
                f.write(f"| {row['phenomenon']} | {row['UID']} | {row['count']} | {row['acc_small']:.2%} | {row['acc_large']:.2%} | {row['delta']:+.2%} |\n")
        f.write("\n")
        
        f.write("### 倒退最大的UIDs:\n")
        top_regressions = df_uid.tail(5)
        if len(top_regressions) > 0:
            f.write("| 现象 | UID | 样本数 | 小模型准确率 | 大模型准确率 | 提升 |\n")
            f.write("|------|-----|--------|-------------|-------------|------|\n")
            for _, row in top_regressions.iterrows():
                f.write(f"| {row['phenomenon']} | {row['UID']} | {row['count']} | {row['acc_small']:.2%} | {row['acc_large']:.2%} | {row['delta']:+.2%} |\n")
        f.write("\n")
        
        f.write("## 6. 结论\n\n")
        f.write("根据分析结果，可以得出以下结论：\n\n")
        f.write(f"1. 大模型在总体准确率上相比小模型提升了 {df_phenom.iloc[:, 1].mean() - df_phenom.iloc[:, 0].mean():.2%}。\n")
        f.write(f"2. McNemar检验的p值为 {p_val:.4e}，{'表明' if p_val < config['analysis']['significant_p_value'] else '表明不'}显著。\n")
        f.write("3. 在不同现象类别中，大模型的表现有提升也有下降，需要进一步分析原因。\n")
        f.write("4. 建议重点关注倒退的UID，分析大模型性能下降的原因。\n")
    
    print(f"📊 详细分析报告已生成: {report_path}")

# 主执行逻辑
if __name__ == "__main__":
    df_05 = load_results_optimized(config["file_paths"]["small_model"])
    df_15 = load_results_optimized(config["file_paths"]["large_model"])

    if df_05 is None or df_15 is None:
        print("❌ 无法加载数据，分析终止")
    else:
        # 1. 基础对比
        df_merged, df_phenom_stats, df_consistency = analyze_comparison(df_05, df_15)
        
        # 2. UID 粒度分析
        df_uid_stats = analyze_uid_stats(df_merged)
        
        # 3. 详细统计
        detailed_stats = analyze_detailed_statistics(df_merged)
        
        # 4. 生成报告
        generate_detailed_report(df_merged, df_phenom_stats, df_consistency, df_uid_stats, 
                               detailed_stats)
        
        # 5. 保存文件
        # 保存合并后的原始数据
        df_merged.to_csv(config['output']['comparison_csv'], 
                        index=False, encoding='utf-8-sig')
        # 保存 UID 统计表
        df_uid_stats.to_csv(config['output']['uid_breakdown_csv'], 
                           index=False, encoding='utf-8-sig')
        
        print("\n💾 文件保存成功:")
        print(f"  1. {config['output']['comparison_csv']} (所有样本详情)")
        print(f"  2. {config['output']['uid_breakdown_csv']} (各 UID 准确率统计表)")
        print(f"  3. {config['output']['visualization_dir']}/analysis_report.md (详细分析报告)")