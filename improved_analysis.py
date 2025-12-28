import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2
import argparse
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ModelComparisonAnalyzer:
    """
    用于比较两个模型性能的分析器
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化分析器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # 默认配置
            self.config = {
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
        
        # 确保输出目录存在
        os.makedirs(self.config["output"]["visualization_dir"], exist_ok=True)
    
    def load_results_optimized(self, filepath: str) -> pd.DataFrame:
        """
        加载模型结果文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的数据框
        """
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

    def calculate_mcnemar(self, matrix_df: pd.DataFrame) -> Tuple[float, float]:
        """
        计算McNemar检验
        
        Args:
            matrix_df: 一致性矩阵
            
        Returns:
            统计量和p值
        """
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

    def analyze_comparison(self, df_small: pd.DataFrame, df_large: pd.DataFrame, 
                          name_s: str = "0.5B", name_l: str = "1.5B") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        比较两个模型的性能
        
        Args:
            df_small: 小模型结果数据框
            df_large: 大模型结果数据框
            name_s: 小模型名称
            name_l: 大模型名称
            
        Returns:
            合并数据框, 现象统计, 一致性矩阵
        """
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
        stat, p_val = self.calculate_mcnemar(df_matrix)
        print(f"\n🧩 [显著性分析] McNemar p-value: {p_val:.4e}")
        print(f"  • 显著性: {'✅ 显著' if p_val < self.config['analysis']['significant_p_value'] else '❌ 不显著'}")

        # 5. Phenomenon 统计
        phenom_group = df_compare.groupby('phenomenon')[[col_s, col_l]].mean()
        phenom_group['Delta'] = phenom_group[col_l] - phenom_group[col_s]
        phenom_group = phenom_group.sort_values('Delta', ascending=False)

        return df_compare, phenom_group, df_matrix

    def analyze_uid_stats(self, df_compare: pd.DataFrame, name_s: str = "0.5B", name_l: str = "1.5B") -> pd.DataFrame:
        """
        计算每个 UID 的详细准确率和 Delta
        
        Args:
            df_compare: 比较数据框
            name_s: 小模型名称
            name_l: 大模型名称
            
        Returns:
            UID 统计数据框
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
        min_samples = self.config['analysis']['min_samples_per_uid']
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

    def analyze_detailed_statistics(self, df_compare: pd.DataFrame, name_s: str = "0.5B", name_l: str = "1.5B") -> Dict:
        """
        计算详细统计信息
        
        Args:
            df_compare: 比较数据框
            name_s: 小模型名称
            name_l: 大模型名称
            
        Returns:
            详细统计信息字典
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

    def visualize_results(self, df_phenom: pd.DataFrame, df_matrix: pd.DataFrame, df_uid: pd.DataFrame, stats: Dict):
        """
        可视化分析结果
        
        Args:
            df_phenom: 现象统计
            df_matrix: 一致性矩阵
            df_uid: UID统计
            stats: 详细统计信息
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型比较分析结果', fontsize=16, fontweight='bold')
        
        # 1. 现象级别的准确率提升
        colors = ['#ff7675' if x < 0 else '#55efc4' for x in df_phenom['Delta']]
        bars = sns.barplot(x=df_phenom['Delta'], y=df_phenom.index, ax=axes[0,0], 
                          hue=df_phenom.index, palette=colors, legend=False)
        axes[0,0].set_title('各现象准确率提升情况')
        axes[0,0].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[0,0].set_xlabel('准确率提升 (大模型 - 小模型)')
        
        # 2. 一致性分布甜甜圈图
        colors_pie = ['#dfe6e9', '#2d3436', '#00b894', '#d63031']
        wedges, texts, autotexts = axes[0,1].pie(df_matrix['Count'], 
                                                labels=df_matrix['Category'], 
                                                autopct='%1.1f%%', 
                                                startangle=140, 
                                                colors=colors_pie, 
                                                wedgeprops=dict(width=0.4))
        axes[0,1].set_title('预测一致性分布')
        
        # 3. UID级别的提升分布
        axes[1,0].hist(df_uid['delta'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].set_title('UID级别准确率提升分布')
        axes[1,0].set_xlabel('准确率提升')
        axes[1,0].set_ylabel('UID数量')
        axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='无变化基准线')
        axes[1,0].legend()
        
        # 4. 整体统计摘要
        axes[1,1].axis('off')
        summary_text = f"""
        📊 整体统计摘要:
        
        总样本数: {stats['total_samples']:,}
        
        准确率:
        • 小模型: {df_phenom.iloc[:, 0].mean():.2%}
        • 大模型: {df_phenom.iloc[:, 1].mean():.2%}
        • 提升: {df_phenom.iloc[:, 1].mean() - df_phenom.iloc[:, 0].mean():+.2%}
        
        预测一致性: {stats['agreement_rate']:.2%}
        显著提升: {stats['improvement_rate']:.2%}
        显著倒退: {stats['regression_rate']:.2%}
        
        现象类别数: {len(df_phenom)}
        """
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f"{self.config['output']['visualization_dir']}/model_comparison_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def generate_detailed_report(self, df_compare: pd.DataFrame, df_phenom: pd.DataFrame, 
                                df_matrix: pd.DataFrame, df_uid: pd.DataFrame, stats: Dict, 
                                name_s: str = "0.5B", name_l: str = "1.5B"):
        """
        生成详细分析报告
        
        Args:
            df_compare: 比较数据框
            df_phenom: 现象统计
            df_matrix: 一致性矩阵
            df_uid: UID统计
            stats: 详细统计信息
            name_s: 小模型名称
            name_l: 大模型名称
        """
        report_path = f"{self.config['output']['visualization_dir']}/analysis_report.md"
        
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
            stat, p_val = self.calculate_mcnemar(df_matrix)
            f.write(f"- McNemar检验 p-value: {p_val:.4e}\n")
            f.write(f"- 显著性: {'显著' if p_val < self.config['analysis']['significant_p_value'] else '不显著'}\n\n")
            
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
            f.write(f"2. McNemar检验的p值为 {p_val:.4e}，{'表明' if p_val < self.config['analysis']['significant_p_value'] else '表明不'}显著。\n")
            f.write("3. 在不同现象类别中，大模型的表现有提升也有下降，需要进一步分析原因。\n")
            f.write("4. 建议重点关注倒退的UID，分析大模型性能下降的原因。\n")
        
        print(f"📊 详细分析报告已生成: {report_path}")

    def run_analysis(self, name_s: str = "0.5B", name_l: str = "1.5B"):
        """
        运行完整分析流程
        
        Args:
            name_s: 小模型名称
            name_l: 大模型名称
        """
        # 加载数据
        df_05 = self.load_results_optimized(self.config["file_paths"]["small_model"])
        df_15 = self.load_results_optimized(self.config["file_paths"]["large_model"])

        if df_05 is None or df_15 is None:
            print("❌ 无法加载数据，分析终止")
            return

        # 1. 基础对比
        df_merged, df_phenom_stats, df_consistency = self.analyze_comparison(df_05, df_15, name_s, name_l)
        
        # 2. UID 粒度分析
        df_uid_stats = self.analyze_uid_stats(df_merged, name_s, name_l)
        
        # 3. 详细统计
        detailed_stats = self.analyze_detailed_statistics(df_merged, name_s, name_l)
        
        # 4. 可视化
        self.visualize_results(df_phenom_stats, df_consistency, df_uid_stats, detailed_stats)
        
        # 5. 生成报告
        self.generate_detailed_report(df_merged, df_phenom_stats, df_consistency, df_uid_stats, 
                                   detailed_stats, name_s, name_l)
        
        # 6. 保存文件
        # 保存合并后的原始数据
        df_merged.to_csv(self.config['output']['comparison_csv'], 
                        index=False, encoding='utf-8-sig')
        # 保存 UID 统计表
        df_uid_stats.to_csv(self.config['output']['uid_breakdown_csv'], 
                           index=False, encoding='utf-8-sig')
        
        print("\n💾 文件保存成功:")
        print(f"  1. {self.config['output']['comparison_csv']} (所有样本详情)")
        print(f"  2. {self.config['output']['uid_breakdown_csv']} (各 UID 准确率统计表)")
        print(f"  3. {self.config['output']['visualization_dir']}/model_comparison_analysis.png (可视化图表)")
        print(f"  4. {self.config['output']['visualization_dir']}/analysis_report.md (详细分析报告)")


def main():
    parser = argparse.ArgumentParser(description='模型比较分析工具')
    parser.add_argument('--config', type=str, default=None, 
                        help='配置文件路径')
    parser.add_argument('--small-model', type=str, 
                        default='Qwen2.5-0.5B_result_full_text.jsonl',
                        help='小模型结果文件路径')
    parser.add_argument('--large-model', type=str, 
                        default='Qwen2.5-1.5B_result_full_text.jsonl',
                        help='大模型结果文件路径')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        "file_paths": {
            "small_model": args.small_model,
            "large_model": args.large_model
        },
        "output": {
            "comparison_csv": "analysis_comparison_result.csv",
            "uid_breakdown_csv": "analysis_uid_breakdown.csv",
            "visualization_dir": args.output_dir
        },
        "analysis": {
            "min_samples_per_uid": 10,
            "significant_p_value": 0.05
        }
    }
    
    # 确保输出目录存在
    os.makedirs(config["output"]["visualization_dir"], exist_ok=True)
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # 创建分析器并运行
    analyzer = ModelComparisonAnalyzer()
    analyzer.config = config
    analyzer.run_analysis()


if __name__ == "__main__":
    main()