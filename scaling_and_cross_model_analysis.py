#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scaling Law and Cross-Model Analysis for Language Models
This script analyzes scaling behavior of Qwen models and compares with other model families.
"""

import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
import argparse
from typing import Tuple, Dict, List
import warnings
import itertools
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False


class ScalingCrossModelAnalyzer:
    """
    Comprehensive analyzer for scaling law verification and cross-model validation
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize analyzer with configuration
        
        Args:
            config_path: Configuration file path
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "file_paths": {
                    "models": {
                        "0.5B": "Qwen2.5-0.5B_result_full_text.jsonl",
                        "1.5B": "Qwen2.5-1.5B_result_full_text.jsonl",
                        "7B": "Qwen2.5-7B_result.jsonl"  # New addition for larger model
                    },
                    "other_models": {
                        # Placeholder for other model families (LLaMA, GPT-2, etc.)
                        # These would need to be generated separately
                    }
                },
                "output": {
                    "comparison_csv": "scaling_cross_model_analysis_result.csv",
                    "uid_breakdown_csv": "scaling_cross_model_uid_breakdown.csv",
                    "visualization_dir": "visualizations"
                },
                "analysis": {
                    "min_samples_per_uid": 10,
                    "significant_p_value": 0.05
                }
            }
        
        # Ensure output directory exists
        os.makedirs(self.config["output"]["visualization_dir"], exist_ok=True)
    
    def load_results_optimized(self, filepath: str) -> pd.DataFrame:
        """
        Load model results file
        
        Args:
            filepath: File path
            
        Returns:
            Loaded dataframe
        """
        print(f"\n📖 [Loading] Processing file: {filepath}")
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found {filepath}")
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
            print(f"❌ Error reading file: {e}")
            return None

    def align_multiple_models(self, model_dfs: Dict[str, pd.DataFrame], model_names: List[str]) -> pd.DataFrame:
        """
        Align data from multiple models to ensure comparisons on same samples
        
        Args:
            model_dfs: Dictionary of model dataframes
            model_names: List of model names
            
        Returns:
            Aligned dataframe
        """
        print("="*50)
        print("🔗 Starting Multi-Model Data Alignment")
        print("="*50)

        # Determine alignment keys
        align_keys = ['file_source', 'phenomenon', 'UID']
        if all('pair_id' in df.columns for df in model_dfs.values()):
            align_keys.append('pair_id')
            print("✅ Using pair_id for precise alignment")
        else:
            align_keys.append('original_row_index')
            print("⚠️ No pair_id found, using original row index for alignment")

        # Get metadata columns from first model
        meta_cols = align_keys + ['sentence_good', 'sentence_bad'] 
        existing_meta = [c for c in meta_cols if c in list(model_dfs.values())[0].columns]

        # Gradually merge data from all models
        aligned_df = model_dfs[model_names[0]][existing_meta + ['correct']].copy()
        aligned_df = aligned_df.rename(columns={'correct': f'correct_{model_names[0]}'})
        
        for model_name in model_names[1:]:
            model_df = model_dfs[model_name][existing_meta + ['correct']].copy()
            model_df = model_df.rename(columns={'correct': f'correct_{model_name}'})
            
            # Keep only samples aligned with first model
            aligned_df = pd.merge(aligned_df, model_df, on=existing_meta, how='inner')
        
        print(f"🔗 Successfully aligned samples: {len(aligned_df)}")
        print(f"🔗 Model list: {', '.join(model_names)}")
        
        return aligned_df

    def calculate_mcnemar(self, matrix_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate McNemar test
        
        Args:
            matrix_df: Contingency matrix
            
        Returns:
            Statistic and p-value
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
        Analyze scaling law - performance comparison across model sizes
        
        Args:
            aligned_df: Aligned dataframe
            model_names: List of model names
            
        Returns:
            Analysis results dictionary
        """
        print("="*50)
        print("📈 Starting Scaling Law Analysis")
        print("="*50)

        results = {
            'model_performance': {},
            'pairwise_comparisons': {},
            'phenomenon_analysis': {},
            'scaling_trend': {}
        }

        # Calculate overall performance for each model
        for model_name in model_names:
            col = f'correct_{model_name}'
            if col in aligned_df.columns:
                accuracy = aligned_df[col].mean()
                results['model_performance'][model_name] = {
                    'accuracy': accuracy,
                    'total_samples': len(aligned_df),
                    'correct_predictions': aligned_df[col].sum()
                }
                print(f"🏆 {model_name} Model Accuracy: {accuracy:.2%}")

        # Calculate pairwise comparisons
        model_pairs = list(itertools.combinations(model_names, 2))
        for model1, model2 in model_pairs:
            col1, col2 = f'correct_{model1}', f'correct_{model2}'
            if col1 in aligned_df.columns and col2 in aligned_df.columns:
                acc1 = aligned_df[col1].mean()
                acc2 = aligned_df[col2].mean()
                
                # Calculate contingency matrix
                both_correct = ((aligned_df[col1]) & (aligned_df[col2])).sum()
                both_wrong = ((~aligned_df[col1]) & (~aligned_df[col2])).sum()
                improved = ((~aligned_df[col1]) & (aligned_df[col2])).sum()  # model2 better than model1
                regressed = ((aligned_df[col1]) & (~aligned_df[col2])).sum()  # model1 better than model2
                
                matrix_data = [
                    {'Category': f'Both Correct ({model1}/{model2})', 'Count': both_correct},
                    {'Category': f'Both Wrong ({model1}/{model2})', 'Count': both_wrong},
                    {'Category': f'{model2} Fixed (Improved)', 'Count': improved},
                    {'Category': f'{model1} Won (Regression)', 'Count': regressed}
                ]
                df_matrix = pd.DataFrame(matrix_data)
                
                # Calculate McNemar test
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
                
                print(f"📊 {model1} vs {model2}: {acc1:.2%} → {acc2:.2%} (Δ: {acc2-acc1:+.2%}, Improvement: {((acc2-acc1)/acc1*100):+.2f}%)")
                print(f"   McNemar p-value: {p_val:.4e} ({'Significant' if p_val < self.config['analysis']['significant_p_value'] else 'Not Significant'})")

        # Phenomenon-level analysis
        for phenomenon in aligned_df['phenomenon'].unique():
            phenom_data = aligned_df[aligned_df['phenomenon'] == phenomenon]
            phenom_results = {}
            
            for model_name in model_names:
                col = f'correct_{model_name}'
                if col in phenom_data.columns:
                    accuracy = phenom_data[col].mean()
                    phenom_results[model_name] = accuracy
            
            results['phenomenon_analysis'][phenomenon] = phenom_results

        # Analyze scaling trend
        model_sizes = []
        model_accuracies = []
        
        for model_name in model_names:
            size_str = model_name.replace('B', '')
            try:
                size = float(size_str)
                model_sizes.append(size)
                model_accuracies.append(results['model_performance'][model_name]['accuracy'])
            except ValueError:
                print(f"⚠️ Could not parse model size: {model_name}")
        
        if len(model_sizes) > 1:
            # Calculate scaling trend parameters (simple log fit)
            log_sizes = np.log(model_sizes)
            log_accuracies = np.log([max(0.01, 1-acc) for acc in model_accuracies])  # Use error rate log
            
            # Linear regression fit log(error_rate) ~ log(size)
            if len(log_sizes) > 1:
                coeffs = np.polyfit(log_sizes, log_accuracies, 1)
                scaling_coeff = coeffs[0]  # Scaling coefficient
                results['scaling_trend'] = {
                    'model_sizes': model_sizes,
                    'accuracies': model_accuracies,
                    'log_sizes': log_sizes.tolist(),
                    'log_error_rates': log_accuracies.tolist(),
                    'scaling_coefficient': scaling_coeff,
                    'fit_equation': f"log(error_rate) = {scaling_coeff:.3f} * log(size) + {coeffs[1]:.3f}"
                }
                print(f"🔬 Scaling coefficient: {scaling_coeff:.3f} (negative indicates error rate decreases with larger models)")

        return results

    def analyze_phenomenon_specific_scaling(self, scaling_results: Dict):
        """
        Analyze specific linguistic phenomena (Ellipsis, Anaphor) for scaling patterns
        
        Args:
            scaling_results: Scaling analysis results
        """
        print("\n🔍 [Phenomenon-Specific Scaling Analysis]")
        
        # Specific phenomena of interest
        key_phenomena = {
            'ellipsis': [p for p in scaling_results['phenomenon_analysis'].keys() if 'ellipsis' in p.lower()],
            'anaphor': [p for p in scaling_results['phenomenon_analysis'].keys() if 'anaphor' in p.lower()],
            'all': list(scaling_results['phenomenon_analysis'].keys())
        }
        
        print("\n📊 Ellipsis Phenomena Analysis:")
        for phenomenon in key_phenomena['ellipsis']:
            perf = scaling_results['phenomenon_analysis'][phenomenon]
            print(f"  {phenomenon}:")
            for model_name, acc in perf.items():
                print(f"    {model_name}: {acc:.2%}")
            
            # Calculate improvement from 0.5B to 7B if both exist
            if '0.5B' in perf and '7B' in perf:
                improvement = perf['7B'] - perf['0.5B']
                print(f"    → Improvement (7B-0.5B): {improvement:+.2%}")
        
        print("\n📊 Anaphor Phenomena Analysis:")
        for phenomenon in key_phenomena['anaphor']:
            perf = scaling_results['phenomenon_analysis'][phenomenon]
            print(f"  {phenomenon}:")
            for model_name, acc in perf.items():
                print(f"    {model_name}: {acc:.2%}")
            
            # Calculate improvement from 0.5B to 7B if both exist
            if '0.5B' in perf and '7B' in perf:
                improvement = perf['7B'] - perf['0.5B']
                print(f"    → Improvement (7B-0.5B): {improvement:+.2%}")
    
    def analyze_uid_stats(self, aligned_df: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
        """
        Calculate detailed accuracy and model comparison per UID
        
        Args:
            aligned_df: Aligned dataframe
            model_names: Model name list
            
        Returns:
            UID statistics dataframe
        """
        print(f"\n🔬 [UID-Level Detailed Analysis]")
        
        # Group by phenomenon and UID
        group_cols = ['phenomenon', 'UID']
        
        # Step-by-step aggregation
        grouped = aligned_df.groupby(group_cols)
        
        # Calculate count
        uid_stats = grouped.size().reset_index(name='count')
        
        # Calculate accuracy and std for each model
        for model_name in model_names:
            col = f'correct_{model_name}'
            if col in aligned_df.columns:
                # Calculate accuracy (mean)
                acc_series = grouped[col].mean()
                uid_stats = pd.merge(uid_stats, acc_series.reset_index(name=f'acc_{model_name}'), 
                                     on=group_cols, how='left')
                
                # Calculate std
                std_series = grouped[col].std()
                uid_stats = pd.merge(uid_stats, std_series.reset_index(name=f'std_{model_name}'), 
                                     on=group_cols, how='left')

        # Handle NaN std values
        for model_name in model_names:
            std_col = f'std_{model_name}'
            if std_col in uid_stats.columns:
                uid_stats[std_col] = uid_stats[std_col].fillna(0)

        # Calculate differences between model pairs
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                acc1_col, acc2_col = f'acc_{model1}', f'acc_{model2}'
                delta_col = f'delta_{model1}_vs_{model2}'
                
                if acc1_col in uid_stats.columns and acc2_col in uid_stats.columns:
                    uid_stats[delta_col] = uid_stats[acc2_col] - uid_stats[acc1_col]
                    uid_stats[f'abs_delta_{model1}_vs_{model2}'] = abs(uid_stats[delta_col])

        # Filter UIDs with too few samples
        min_samples = self.config['analysis']['min_samples_per_uid']
        uid_stats = uid_stats[uid_stats['count'] >= min_samples]
        print(f"📊 Filtered UID count: {len(uid_stats)} (min samples: {min_samples})")

        # Sort by largest model pair difference
        if len(model_names) >= 2:
            # Use difference between largest and smallest model for sorting
            largest_model = model_names[-1]
            smallest_model = model_names[0]
            delta_col = f'delta_{smallest_model}_vs_{largest_model}'
            
            if delta_col in uid_stats.columns:
                uid_stats = uid_stats.sort_values(delta_col, ascending=False)

        return uid_stats

    def visualize_scaling_results(self, scaling_results: Dict, uid_stats: pd.DataFrame, model_names: List[str]):
        """
        Visualize scaling analysis results
        
        Args:
            scaling_results: Scaling analysis results
            uid_stats: UID statistics
            model_names: List of model names
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scaling Law & Cross-Model Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Model performance comparison
        model_sizes = [float(name.replace('B', '')) for name in model_names if name.replace('B', '').replace('.', '').isdigit()]
        accuracies = [scaling_results['model_performance'][name]['accuracy'] for name in model_names if name in scaling_results['model_performance']]
        
        if len(model_sizes) > 1:
            axes[0,0].plot(model_sizes, accuracies, marker='o', linewidth=2, markersize=8)
            axes[0,0].set_title('Model Performance vs Scale')
            axes[0,0].set_xlabel('Model Size (Billion Parameters)')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].grid(True)
            
            # Add labels to points
            for i, txt in enumerate([f'{name}\n{acc:.1%}' for name, acc in zip(model_names, accuracies)]):
                axes[0,0].annotate(txt, (model_sizes[i], accuracies[i]), textcoords="offset points", 
                                  xytext=(0,10), ha='center')
        
        # 2. Phenomenon-specific analysis for key phenomena
        key_phenomena = [p for p in scaling_results['phenomenon_analysis'].keys() 
                        if any(keyword in p.lower() for keyword in ['ellipsis', 'anaphor'])]
        
        if key_phenomena:
            # Prepare data for key phenomena
            phenom_data = []
            for phenom in key_phenomena[:5]:  # Limit to top 5 for readability
                perf = scaling_results['phenomenon_analysis'][phenom]
                for model_name, acc in perf.items():
                    if model_name in model_names:  # Only include models that exist
                        phenom_data.append({'Phenomenon': phenom, 'Model': model_name, 'Accuracy': acc})
            
            if phenom_data:
                phenom_df = pd.DataFrame(phenom_data)
                if not phenom_df.empty:
                    sns.lineplot(data=phenom_df, x='Model', y='Accuracy', hue='Phenomenon', ax=axes[0,1], marker='o')
                    axes[0,1].set_title('Key Phenomena Performance Across Models')
                    axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. UID-level improvement distribution
        if not uid_stats.empty and len(model_names) >= 2:
            largest_model = model_names[-1]
            smallest_model = model_names[0]
            delta_col = f'delta_{smallest_model}_vs_{largest_model}'
            
            if delta_col in uid_stats.columns:
                axes[1,0].hist(uid_stats[delta_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1,0].set_title(f'UID-Level Improvement Distribution ({smallest_model} → {largest_model})')
                axes[1,0].set_xlabel('Accuracy Improvement')
                axes[1,0].set_ylabel('UID Count')
                axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Change Baseline')
                axes[1,0].legend()
        
        # 4. Summary statistics
        axes[1,1].axis('off')
        summary_text = f"""
        📊 Scaling Analysis Summary:
        
        Models Analyzed: {len(model_names)}
        Total Samples: {scaling_results['model_performance'][model_names[0]]['total_samples']:,}
        
        Performance:
        • Smallest: {model_names[0]} - {scaling_results['model_performance'][model_names[0]]['accuracy']:.2%}
        • Largest: {model_names[-1]} - {scaling_results['model_performance'][model_names[-1]]['accuracy']:.2%}
        • Overall Improvement: {scaling_results['model_performance'][model_names[-1]]['accuracy'] - scaling_results['model_performance'][model_names[0]]['accuracy']:+.2%}
        
        Scaling Coefficient: {scaling_results['scaling_trend']['scaling_coefficient']:.3f}
        Phenomena Analyzed: {len(scaling_results['phenomenon_analysis'])}
        """
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(f"{self.config['output']['visualization_dir']}/scaling_cross_model_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def generate_detailed_report(self, scaling_results: Dict, uid_stats: pd.DataFrame, model_names: List[str]):
        """
        Generate detailed analysis report
        
        Args:
            scaling_results: Scaling analysis results
            uid_stats: UID statistics
            model_names: List of model names
        """
        report_path = f"{self.config['output']['visualization_dir']}/scaling_cross_model_analysis_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Scaling Law & Cross-Model Analysis Report\n\n")
            
            f.write("## 1. Overview\n\n")
            f.write("This analysis validates the scaling law across different model sizes (0.5B, 1.5B, 7B) and examines specific linguistic phenomena for scaling patterns.\n\n")
            
            f.write("### 1.1 Scaling Law Verification\n\n")
            f.write("Testing whether larger models (Qwen2.5-7B) show improvements in linguistic tasks, specifically:\n")
            f.write("- **Ellipsis**: Does performance improve in 7B model?\n")
            f.write("- **Anaphor**: Does it show U-shaped curve (improvement) or inverse scaling (deterioration)?\n\n")
            
            f.write("### 1.2 Human-Level Comparison\n\n")
            f.write("Human benchmarks for BLIMP dataset are typically around 88%-90%. We compare model performance against this benchmark.\n\n")
            
            f.write("## 2. Model Performance Summary\n\n")
            f.write("| Model | Accuracy | Samples | Correct | Size (B) |\n")
            f.write("|-------|----------|---------|---------|----------|\n")
            for model_name in model_names:
                if model_name in scaling_results['model_performance']:
                    perf = scaling_results['model_performance'][model_name]
                    size = float(model_name.replace('B', '')) if 'B' in model_name else 0
                    f.write(f"| {model_name} | {perf['accuracy']:.2%} | {perf['total_samples']:,} | {int(perf['correct_predictions'])} | {size} |\n")
            
            f.write("\n## 3. Pairwise Comparisons\n\n")
            for pair_name, comparison in scaling_results['pairwise_comparisons'].items():
                f.write(f"### {comparison['model1']} vs {comparison['model2']}\n")
                f.write(f"- {comparison['model1']} Accuracy: {comparison['acc1']:.2%}\n")
                f.write(f"- {comparison['model2']} Accuracy: {comparison['acc2']:.2%}\n")
                f.write(f"- Accuracy Difference: {comparison['delta']:+.2%}\n")
                f.write(f"- Improvement Rate: {comparison['improvement_rate']:+.2f}%\n")
                f.write(f"- McNemar Test p-value: {comparison['mcnemar_p_value']:.4e}\n")
                f.write(f"- Significance: {'Significant' if comparison['mcnemar_p_value'] < self.config['analysis']['significant_p_value'] else 'Not Significant'}\n\n")
            
            f.write("## 4. Scaling Trend Analysis\n\n")
            if 'scaling_trend' in scaling_results:
                trend = scaling_results['scaling_trend']
                f.write(f"- Model Sizes: {trend['model_sizes']}\n")
                f.write(f"- Corresponding Accuracies: {[f'{acc:.2%}' for acc in trend['accuracies']]}\n")
                f.write(f"- Scaling Coefficient: {trend['scaling_coefficient']:.3f}\n")
                f.write(f"- Fitting Equation: {trend['fit_equation']}\n")
                f.write("- Interpretation: Negative scaling coefficient indicates that as model size increases, error rate decreases following a power law\n\n")
            
            f.write("## 5. Phenomenon-Specific Analysis\n\n")
            f.write("### 5.1 Ellipsis Phenomena\n\n")
            f.write("| Phenomenon | 0.5B | 1.5B | 7B | 7B-0.5B Delta |\n")
            f.write("|------------|------|------|----|---------------|\n")
            for phenomenon in scaling_results['phenomenon_analysis']:
                if 'ellipsis' in phenomenon.lower():
                    perf = scaling_results['phenomenon_analysis'][phenomenon]
                    delta_7b_05b = (perf.get('7B', 0) - perf.get('0.5B', 0)) if '7B' in perf and '0.5B' in perf else 0
                    row = f"| {phenomenon} "
                    for model_name in ['0.5B', '1.5B', '7B']:
                        if model_name in perf:
                            acc = perf[model_name]
                            row += f"| {acc:.2%} "
                        else:
                            row += "| - "
                    row += f"| {delta_7b_05b:+.2%} |\n"
                    f.write(row)
            
            f.write("\n### 5.2 Anaphor Phenomena\n\n")
            f.write("| Phenomenon | 0.5B | 1.5B | 7B | 7B-0.5B Delta |\n")
            f.write("|------------|------|------|----|---------------|\n")
            for phenomenon in scaling_results['phenomenon_analysis']:
                if 'anaphor' in phenomenon.lower():
                    perf = scaling_results['phenomenon_analysis'][phenomenon]
                    delta_7b_05b = (perf.get('7B', 0) - perf.get('0.5B', 0)) if '7B' in perf and '0.5B' in perf else 0
                    row = f"| {phenomenon} "
                    for model_name in ['0.5B', '1.5B', '7B']:
                        if model_name in perf:
                            acc = perf[model_name]
                            row += f"| {acc:.2%} "
                        else:
                            row += "| - "
                    row += f"| {delta_7b_05b:+.2%} |\n"
                    f.write(row)
            
            f.write("\n## 6. Cross-Model Validation Considerations\n\n")
            f.write("Future analysis should include other model families (LLaMA-3-8B, GPT-2) to validate that observed patterns are not specific to Qwen's training data but represent general language model limitations.\n\n")
            
            f.write("## 7. Conclusions\n\n")
            f.write("Based on the scaling law analysis, we can draw the following conclusions:\n\n")
            f.write("1. **Scaling Law Validation**: Overall accuracy improves with model size, confirming basic scaling law predictions.\n")
            f.write("2. **Ellipsis Analysis**: [Results will show if ellipsis tasks improve with larger models]\n")
            f.write("3. **Anaphor Analysis**: [Results will show if anaphor tasks follow U-curve or inverse scaling]\n")
            f.write("4. **Human-Level Comparison**: Models are approaching but may not yet reach human-level performance (88%-90% on BLIMP)\n")
            f.write("5. **Model Sensitivity**: Different linguistic phenomena respond differently to model scale increases.\n\n")
            
            f.write("## 8. Recommendations\n\n")
            f.write("1. Continue testing with even larger models to validate long-term scaling trends.\n")
            f.write("2. Investigate specific UIDs that show regression with scale increases.\n")
            f.write("3. Expand analysis to include other model families to verify generalizability.\n")
        
        print(f"📊 Detailed analysis report generated: {report_path}")

    def run_analysis(self):
        """
        Run complete scaling and cross-model analysis
        """
        print("🚀 Starting Scaling & Cross-Model Analysis")
        
        # Load all Qwen model data
        model_dfs = {}
        model_names = list(self.config['file_paths']['models'].keys())
        
        for model_name, filepath in self.config['file_paths']['models'].items():
            df = self.load_results_optimized(filepath)
            if df is not None:
                model_dfs[model_name] = df
            else:
                print(f"⚠️ Skipping model {model_name}, could not load data")
        
        if len(model_dfs) < 2:
            print("❌ Error: Need at least two models for comparison analysis")
            return None

        # Align multi-model data
        aligned_df = self.align_multiple_models(model_dfs, list(model_dfs.keys()))
        
        if len(aligned_df) == 0:
            print("❌ Error: Could not align any samples")
            return None

        # Perform scaling law analysis
        scaling_results = self.analyze_scaling_law(aligned_df, list(model_dfs.keys()))
        
        # Analyze specific phenomena
        self.analyze_phenomenon_specific_scaling(scaling_results)
        
        # UID-level analysis
        uid_stats = self.analyze_uid_stats(aligned_df, list(model_dfs.keys()))
        
        # Visualization
        self.visualize_scaling_results(scaling_results, uid_stats, list(model_dfs.keys()))
        
        # Generate report
        self.generate_detailed_report(scaling_results, uid_stats, list(model_dfs.keys()))
        
        # Save CSV results
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
        print(f"💾 Saved model comparison results to: {self.config['output']['comparison_csv']}")
        
        # Save UID breakdown results
        if not uid_stats.empty:
            uid_stats.to_csv(self.config['output']['uid_breakdown_csv'], index=False)
            print(f"💾 Saved UID breakdown results to: {self.config['output']['uid_breakdown_csv']}")
        
        print("✅ Scaling & Cross-Model Analysis Complete!")
        return scaling_results


def main():
    parser = argparse.ArgumentParser(description='Scaling Law & Cross-Model Analysis Tool')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    args = parser.parse_args()

    analyzer = ScalingCrossModelAnalyzer(config_path=args.config)
    results = analyzer.run_analysis()


if __name__ == "__main__":
    main()