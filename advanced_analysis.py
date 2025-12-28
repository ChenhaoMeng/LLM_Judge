import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2
import argparse
from typing import Tuple, Dict, List, Optional
import warnings
import itertools
warnings.filterwarnings('ignore')

class ScalingLawAnalyzer:
    """
    用于验证缩放定律的分析器，支持多个模型的比较
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
                    "models": {
                        "0.5B": "Qwen2.5-0.5B_result_full_text.jsonl",
                        "1.5B": "Qwen2.5-1.5B_result_full_text.jsonl",
                        "7B": "Qwen2.5-7B_result.jsonl"  # 新增7B模型
                    }
                },
                "output": {
                    "comparison_csv": "scaling_law_analysis_result.csv",
                    "uid_breakdown_csv": "scaling_law_uid_breakdown.csv",
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

    def align_multiple_models(self, model_dfs: Dict[str, pd.DataFrame], model_names: List[str]) -> pd.DataFrame:
        """
        对齐多个模型的数据，确保它们在相同的样本上进行比较

        Args:
            model_dfs: 模型数据框字典
            model_names: 模型名称列表

        Returns:
            对齐后的数据框
        """
        print("="*50)
        print("🔗 开始多模型数据对齐")
        print("="*50)

        # 确定对齐键
        align_keys = ['file_source', 'phenomenon', 'UID']
        if all('pair_id' in df.columns for df in model_dfs.values()):
            align_keys.append('pair_id')
            print("✅ 使用 pair_id 进行精确对齐")
        else:
            align_keys.append('original_row_index')
            print("⚠️ 未发现 pair_id，使用原始行号 (row_index) 对齐")

        # 获取所有模型的元数据列
        meta_cols = align_keys + ['sentence_good', 'sentence_bad'] 
        existing_meta = [c for c in meta_cols if c in list(model_dfs.values())[0].columns]

        # 逐步合并所有模型的数据
        aligned_df = model_dfs[model_names[0]][existing_meta + ['correct']].copy()
        aligned_df = aligned_df.rename(columns={'correct': f'correct_{model_names[0]}'})
        
        for model_name in model_names[1:]:
            model_df = model_dfs[model_name][existing_meta + ['correct']].copy()
            model_df = model_df.rename(columns={'correct': f'correct_{model_name}'})
            
            # 只保留与第一个模型对齐的样本
            aligned_df = pd.merge(aligned_df, model_df, on=existing_meta, how='inner')
        
        print(f"🔗 成功对齐样本数: {len(aligned_df)}")
        print(f"🔗 模型列表: {', '.join(model_names)}")
        
        return aligned_df

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

    def analyze_scaling_law(self, aligned_df: pd.DataFrame, model_names: List[str]) -> Dict:
        """
        分析缩放定律，计算不同模型之间的性能比较

        Args:
            aligned_df: 对齐后的数据框
            model_names: 模型名称列表

        Returns:
            分析结果字典
        """
        print("="*50)
        print("📈 开始缩放定律分析")
        print("="*50)

        results = {
            'model_performance': {},
            'pairwise_comparisons': {},
            'phenomenon_analysis': {},
            'scaling_trend': {}
        }

        # 计算每个模型的总体性能
        for model_name in model_names:
            col = f'correct_{model_name}'
            if col in aligned_df.columns:
                accuracy = aligned_df[col].mean()
                results['model_performance'][model_name] = {
                    'accuracy': accuracy,
                    'total_samples': len(aligned_df),
                    'correct_predictions': aligned_df[col].sum()
                }
                print(f"🏆 {model_name} 模型准确率: {accuracy:.2%}")

        # 计算模型对之间的比较
        model_pairs = list(itertools.combinations(model_names, 2))
        for model1, model2 in model_pairs:
            col1, col2 = f'correct_{model1}', f'correct_{model2}'
            if col1 in aligned_df.columns and col2 in aligned_df.columns:
                acc1 = aligned_df[col1].mean()
                acc2 = aligned_df[col2].mean()
                
                # 计算一致性矩阵
                both_correct = ((aligned_df[col1]) & (aligned_df[col2])).sum()
                both_wrong = ((~aligned_df[col1]) & (~aligned_df[col2])).sum()
                improved = ((~aligned_df[col1]) & (aligned_df[col2])).sum()  # model2比model1好
                regressed = ((aligned_df[col1]) & (~aligned_df[col2])).sum()  # model1比model2好
                
                matrix_data = [
                    {'Category': f'Both Correct ({model1}/{model2})', 'Count': both_correct},
                    {'Category': f'Both Wrong ({model1}/{model2})', 'Count': both_wrong},
                    {'Category': f'{model2} Fixed (Improved)', 'Count': improved},
                    {'Category': f'{model1} Won (Regression)', 'Count': regressed}
                ]
                df_matrix = pd.DataFrame(matrix_data)
                
                # 计算McNemar检验
                stat, p_val = self.calculate_mcnemar(df_matrix)
                
                results['pairwise_comparisons'][f"{model1}_vs_{model2}"] = {
                    'model1': model1,
                    'model2': model2,
                    'acc1': acc1,
                    'acc2': acc2,
                    'delta': acc2 - acc1,
                    'improvement_rate': (acc2 - acc1) / acc1 * 100 if acc1 != 0 else 0,
                    'improved_samples': improved,
                    'regressed_samples': regressed,
                    'mcnemar_stat': stat,
                    'mcnemar_p_value': p_val,
                    'matrix': df_matrix
                }
                
                print(f"📊 {model1} vs {model2}: {acc1:.2%} → {acc2:.2%} (Δ: {acc2-acc1:+.2%}, 提升: {((acc2-acc1)/acc1*100):+.2f}%)")
                print(f"   McNemar p-value: {p_val:.4e} ({'显著' if p_val < self.config['analysis']['significant_p_value'] else '不显著'})")

        # 现象级别的分析
        for phenomenon in aligned_df['phenomenon'].unique():
            phenom_data = aligned_df[aligned_df['phenomenon'] == phenomenon]
            phenom_results = {}
            
            for model_name in model_names:
                col = f'correct_{model_name}'
                if col in phenom_data.columns:
                    accuracy = phenom_data[col].mean()
                    phenom_results[model_name] = accuracy
            
            results['phenomenon_analysis'][phenomenon] = phenom_results

        # 分析缩放趋势
        model_sizes = []
        model_accuracies = []
        
        for model_name in model_names:
            size_str = model_name.replace('B', '')
            try:
                size = float(size_str)
                model_sizes.append(size)
                model_accuracies.append(results['model_performance'][model_name]['accuracy'])
            except ValueError:
                print(f"⚠️ 无法解析模型大小: {model_name}")
        
        if len(model_sizes) > 1:
            # 计算缩放趋势参数 (简单的对数拟合)
            log_sizes = np.log(model_sizes)
            log_accuracies = np.log([max(0.01, 1-acc) for acc in model_accuracies])  # 使用错误率的对数
            
            # 线性回归拟合 log(error_rate) ~ log(size)
            if len(log_sizes) > 1:
                coeffs = np.polyfit(log_sizes, log_accuracies, 1)
                scaling_coeff = coeffs[0]  # 缩放系数
                results['scaling_trend'] = {
                    'model_sizes': model_sizes,
                    'accuracies': model_accuracies,
                    'log_sizes': log_sizes.tolist(),
                    'log_error_rates': log_accuracies.tolist(),
                    'scaling_coefficient': scaling_coeff,
                    'fit_equation': f"log(error_rate) = {scaling_coeff:.3f} * log(size) + {coeffs[1]:.3f}"
                }
                print(f"🔬 缩放系数: {scaling_coeff:.3f} (负值表示模型越大错误率越低)")

        return results

    def analyze_uid_stats(self, aligned_df: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
        """
        计算每个 UID 的详细准确率和模型间比较

        Args:
            aligned_df: 对齐的数据框
            model_names: 模型名称列表

        Returns:
            UID 统计数据框
        """
        print(f"\n🔬 [UID 粒度详细分析]")
        
        # 按 Phenomenon 和 UID 分组聚合
        group_cols = ['phenomenon', 'UID']
        
        # 分步进行聚合
        grouped = aligned_df.groupby(group_cols)
        
        # 计算计数
        uid_stats = grouped.size().reset_index(name='count')
        
        # 为每个模型计算准确率和标准差
        for model_name in model_names:
            col = f'correct_{model_name}'
            if col in aligned_df.columns:
                # 计算准确率（均值）
                acc_series = grouped[col].mean()
                uid_stats = pd.merge(uid_stats, acc_series.reset_index(name=f'acc_{model_name}'), 
                                     on=group_cols, how='left')
                
                # 计算标准差
                std_series = grouped[col].std()
                uid_stats = pd.merge(uid_stats, std_series.reset_index(name=f'std_{model_name}'), 
                                     on=group_cols, how='left')

        # 处理标准差为 NaN 的情况
        for model_name in model_names:
            std_col = f'std_{model_name}'
            if std_col in uid_stats.columns:
                uid_stats[std_col] = uid_stats[std_col].fillna(0)

        # 计算模型对之间的差异
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                acc1_col, acc2_col = f'acc_{model1}', f'acc_{model2}'
                delta_col = f'delta_{model1}_vs_{model2}'
                
                if acc1_col in uid_stats.columns and acc2_col in uid_stats.columns:
                    uid_stats[delta_col] = uid_stats[acc2_col] - uid_stats[acc1_col]
                    uid_stats[f'abs_delta_{model1}_vs_{model2}'] = abs(uid_stats[delta_col])

        # 过滤样本数过少的UID
        min_samples = self.config['analysis']['min_samples_per_uid']
        uid_stats = uid_stats[uid_stats['count'] >= min_samples]
        print(f"📊 过滤后UID数量: {len(uid_stats)} (最小样本数: {min_samples})")

        # 按最大的模型对差异排序
        if len(model_names) >= 2:
            # 使用最大模型与最小模型的差异作为排序依据
            largest_model = model_names[-1]
            smallest_model = model_names[0]
            delta_col = f'delta_{smallest_model}_vs_{largest_model}'
            
            if delta_col in uid_stats.columns:
                uid_stats = uid_stats.sort_values(delta_col, ascending=False)

        return uid_stats

    def visualize_scaling_law(self, scaling_results: Dict, uid_stats: pd.DataFrame):
        """
        可视化缩放定律分析结果

        Args:
            scaling_results: 缩放定律分析结果
            uid_stats: UID统计
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('缩放定律分析结果', fontsize=16, fontweight='bold')

        # 1. 模型性能对比柱状图
        model_names = list(scaling_results['model_performance'].keys())
        accuracies = [scaling_results['model_performance'][name]['accuracy'] for name in model_names]
        
        bars = axes[0,0].bar(model_names, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        axes[0,0].set_title('不同模型准确率对比')
        axes[0,0].set_ylabel('准确率')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 在柱子上显示数值
        for bar, acc in zip(bars, accuracies):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{acc:.1%}', ha='center', va='bottom')

        # 2. 缩放趋势图 (如果可用)
        if 'scaling_trend' in scaling_results:
            trend = scaling_results['scaling_trend']
            axes[0,1].plot(trend['model_sizes'], trend['accuracies'], 'o-', linewidth=2, markersize=8)
            axes[0,1].set_title('模型规模-性能缩放趋势')
            axes[0,1].set_xlabel('模型规模 (B)')
            axes[0,1].set_ylabel('准确率')
            axes[0,1].grid(True, alpha=0.3)
            
            # 添加缩放定律拟合线
            if len(trend['model_sizes']) > 1:
                # 基于缩放系数的拟合
                x_smooth = np.linspace(min(trend['model_sizes']), max(trend['model_sizes']), 100)
                # 使用幂律拟合: accuracy = 1 - C * (size)^k
                # 这里简单地使用对数拟合的逆变换
                log_x = np.log(trend['model_sizes'])
                log_y = np.log([max(0.01, 1-acc) for acc in trend['accuracies']])
                coeffs = np.polyfit(log_x, log_y, 1)
                
                # 使用拟合参数生成平滑曲线
                log_pred = coeffs[0] * np.log(x_smooth) + coeffs[1]
                y_smooth = 1 - np.exp(log_pred)
                axes[0,1].plot(x_smooth, y_smooth, '--', color='red', alpha=0.7, label='缩放定律拟合')
                axes[0,1].legend()

        # 3. 模型对比较热力图
        if scaling_results['pairwise_comparisons']:
            model_pairs = list(scaling_results['pairwise_comparisons'].keys())
            deltas = [scaling_results['pairwise_comparisons'][pair]['delta'] for pair in model_pairs]
            
            # 创建热力图数据
            n_models = len(model_names)
            if n_models >= 2:
                delta_matrix = np.zeros((n_models, n_models))
                pair_labels = []
                
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names):
                        if i != j and f"{model1}_vs_{model2}" in scaling_results['pairwise_comparisons']:
                            delta_matrix[i][j] = scaling_results['pairwise_comparisons'][f"{model1}_vs_{model2}"]['delta']
                        elif i == j:
                            delta_matrix[i][j] = 0.0  # 对角线为0
                
                im = axes[1,0].imshow(delta_matrix, cmap='RdBu_r', aspect='auto')
                axes[1,0].set_xticks(range(n_models))
                axes[1,0].set_yticks(range(n_models))
                axes[1,0].set_xticklabels(model_names)
                axes[1,0].set_yticklabels(model_names)
                axes[1,0].set_title('模型对准确率差异热力图')
                
                # 添加数值标注
                for i in range(n_models):
                    for j in range(n_models):
                        text = axes[1,0].text(j, i, f'{delta_matrix[i, j]:+.1%}', 
                                            ha="center", va="center", color="black" if abs(delta_matrix[i, j]) < 0.05 else "white")
                
                plt.colorbar(im, ax=axes[1,0])

        # 4. UID级别提升分布
        if len(model_names) >= 2:
            largest_model = model_names[-1]
            smallest_model = model_names[0]
            delta_col = f'delta_{smallest_model}_vs_{largest_model}'
            
            if delta_col in uid_stats.columns:
                axes[1,1].hist(uid_stats[delta_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1,1].set_title(f'UID级别准确率提升分布 ({smallest_model} vs {largest_model})')
                axes[1,1].set_xlabel('准确率提升')
                axes[1,1].set_ylabel('UID数量')
                axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='无变化基准线')
                axes[1,1].legend()

        plt.tight_layout()
        plt.savefig(f"{self.config['output']['visualization_dir']}/scaling_law_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def generate_scaling_report(self, scaling_results: Dict, uid_stats: pd.DataFrame):
        """
        生成缩放定律分析报告

        Args:
            scaling_results: 缩放定律分析结果
            uid_stats: UID统计
        """
        report_path = f"{self.config['output']['visualization_dir']}/scaling_law_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 缩放定律分析报告\n\n")
            f.write("## 1. 概述\n\n")
            f.write("本次分析验证了不同规模模型（0.5B, 1.5B, 7B）的缩放定律，探讨模型参数增加对性能的影响。\n\n")

            f.write("## 2. 模型性能汇总\n\n")
            f.write("| 模型 | 准确率 | 样本数 | 正确预测数 |\n")
            f.write("|------|--------|--------|------------|\n")
            for model_name, perf in scaling_results['model_performance'].items():
                f.write(f"| {model_name} | {perf['accuracy']:.2%} | {perf['total_samples']:,} | {int(perf['correct_predictions'])} |\n")

            f.write("\n## 3. 模型对比较\n\n")
            for pair_name, comparison in scaling_results['pairwise_comparisons'].items():
                f.write(f"### {comparison['model1']} vs {comparison['model2']}\n")
                f.write(f"- {comparison['model1']} 准确率: {comparison['acc1']:.2%}\n")
                f.write(f"- {comparison['model2']} 准确率: {comparison['acc2']:.2%}\n")
                f.write(f"- 准确率差异: {comparison['delta']:+.2%}\n")
                f.write(f"- 提升率: {comparison['improvement_rate']:+.2f}%\n")
                f.write(f"- McNemar检验 p-value: {comparison['mcnemar_p_value']:.4e}\n")
                f.write(f"- 显著性: {'显著' if comparison['mcnemar_p_value'] < self.config['analysis']['significant_p_value'] else '不显著'}\n\n")

            f.write("## 4. 缩放趋势分析\n\n")
            if 'scaling_trend' in scaling_results:
                trend = scaling_results['scaling_trend']
                f.write(f"- 模型规模: {trend['model_sizes']}\n")
                f.write(f"- 对应准确率: {[f'{acc:.2%}' for acc in trend['accuracies']]}\n")
                f.write(f"- 缩放系数: {trend['scaling_coefficient']:.3f}\n")
                f.write(f"- 拟合方程: {trend['fit_equation']}\n")
                f.write("- 解释: 缩放系数为负值表示随着模型规模增加，错误率呈幂律下降\n\n")

            f.write("## 5. 现象级别分析\n\n")
            f.write("| 现象 | 0.5B模型 | 1.5B模型 | 7B模型 | 差异 (7B-0.5B) |\n")
            f.write("|------|----------|----------|---------|---------------|\n")
            for phenomenon, perf in list(scaling_results['phenomenon_analysis'].items())[:10]:  # 只显示前10个
                row = f"| {phenomenon} "
                delta_7b_05b = 0
                for model_name in ['0.5B', '1.5B', '7B']:
                    if model_name in perf:
                        acc = perf[model_name]
                        row += f"| {acc:.2%} "
                        if model_name == '7B' and '0.5B' in perf:
                            delta_7b_05b = perf['7B'] - perf['0.5B']
                    else:
                        row += "| - "
                row += f"| {delta_7b_05b:+.2%} |\n"
                f.write(row)

            f.write("\n## 6. 结论\n\n")
            f.write("根据缩放定律分析结果，可以得出以下结论：\n\n")
            f.write("1. 随着模型规模的增加，整体准确率呈现提升趋势，符合缩放定律的基本预测。\n")
            f.write("2. 从小模型到大模型的性能提升在统计上显著（通过McNemar检验验证）。\n")
            f.write("3. 不同语言现象对模型规模的敏感度不同，某些现象在大模型上提升更明显。\n")
            f.write("4. 缩放系数提供了模型性能随规模变化的量化指标。\n")
            f.write("5. 建议继续探索更大规模模型的性能表现，验证缩放定律的长期趋势。\n")

    def run_analysis(self):
        """
        运行完整的缩放定律分析流程
        """
        print("🚀 开始缩放定律分析流程")
        
        # 加载所有模型的数据
        model_dfs = {}
        model_names = list(self.config['file_paths']['models'].keys())
        
        for model_name, filepath in self.config['file_paths']['models'].items():
            df = self.load_results_optimized(filepath)
            if df is not None:
                model_dfs[model_name] = df
            else:
                print(f"⚠️ 跳过模型 {model_name}，因为无法加载数据")
        
        if len(model_dfs) < 2:
            print("❌ 错误: 至少需要两个模型的数据才能进行比较分析")
            return None

        # 对齐多模型数据
        aligned_df = self.align_multiple_models(model_dfs, list(model_dfs.keys()))
        
        if len(aligned_df) == 0:
            print("❌ 错误: 无法对齐任何样本")
            return None

        # 执行缩放定律分析
        scaling_results = self.analyze_scaling_law(aligned_df, list(model_dfs.keys()))
        
        # UID级别分析
        uid_stats = self.analyze_uid_stats(aligned_df, list(model_dfs.keys()))
        
        # 可视化结果
        self.visualize_scaling_law(scaling_results, uid_stats)
        
        # 生成报告
        self.generate_scaling_report(scaling_results, uid_stats)
        
        # 保存CSV结果
        comparison_results = []
        for model_name, perf in scaling_results['model_performance'].items():
            comparison_results.append({
                'Model': model_name,
                'Accuracy': perf['accuracy'],
                'Total_Samples': perf['total_samples'],
                'Correct_Predictions': int(perf['correct_predictions'])
            })
        
        df_comparison = pd.DataFrame(comparison_results)
        df_comparison.to_csv(self.config['output']['comparison_csv'], index=False)
        print(f"💾 已保存模型比较结果到: {self.config['output']['comparison_csv']}")
        
        # 保存UID分解结果
        if not uid_stats.empty:
            uid_stats.to_csv(self.config['output']['uid_breakdown_csv'], index=False)
            print(f"💾 已保存UID分解结果到: {self.config['output']['uid_breakdown_csv']}")
        
        print("✅ 缩放定律分析完成！")
        return scaling_results


def main():
    parser = argparse.ArgumentParser(description='缩放定律分析工具')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    args = parser.parse_args()

    analyzer = ScalingLawAnalyzer(config_path=args.config)
    results = analyzer.run_analysis()

if __name__ == "__main__":
    main()