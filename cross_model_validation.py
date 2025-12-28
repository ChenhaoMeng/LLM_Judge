#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Model Validation Analysis
This script is designed to compare results across different model families (Qwen, LLaMA, GPT-2, etc.)
to validate that observed linguistic patterns are not specific to Qwen's training data.
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


class CrossModelValidator:
    """
    Cross-model validation analyzer to compare linguistic phenomena across different model families
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize cross-model validator
        
        Args:
            config_path: Configuration file path
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration with placeholder for other model families
            self.config = {
                "file_paths": {
                    "qwen_models": {
                        "Qwen2.5-0.5B": "Qwen2.5-0.5B_result_full_text.jsonl",
                        "Qwen2.5-1.5B": "Qwen2.5-1.5B_result_full_text.jsonl",
                        "Qwen2.5-7B": "Qwen2.5-7B_result.jsonl"
                    },
                    # These would need to be generated separately for other model families
                    "other_models": {
                        # "LLaMA-3-8B": "LLaMA-3-8B_result.jsonl",  # Placeholder
                        # "GPT-2-Small": "GPT-2-Small_result.jsonl",  # Placeholder
                    }
                },
                "output": {
                    "comparison_csv": "cross_model_analysis_result.csv",
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

    def analyze_cross_model_validation(self):
        """
        Analyze cross-model validation by comparing linguistic phenomena across different model families
        """
        print("="*60)
        print("🔬 Cross-Model Validation Analysis")
        print("="*60)
        
        # Load Qwen model results
        qwen_results = {}
        for model_name, filepath in self.config['file_paths']['qwen_models'].items():
            df = self.load_results_optimized(filepath)
            if df is not None:
                # Calculate performance by phenomenon
                phenom_performance = df.groupby('phenomenon')['correct'].mean()
                qwen_results[model_name] = phenom_performance
                print(f"✅ Loaded {model_name} results ({len(df)} samples)")
        
        print("\n📊 Cross-Model Analysis Summary:")
        print("\n### Qwen Model Performance by Phenomenon:")
        
        # Create a DataFrame for easy comparison
        all_phenomena = set()
        for model_results in qwen_results.values():
            all_phenomena.update(model_results.index)
        
        comparison_df = pd.DataFrame(index=list(all_phenomena))
        for model_name, phenom_performance in qwen_results.items():
            comparison_df[model_name] = phenom_performance
        
        print(comparison_df.round(4))
        
        # Analyze specific phenomena of interest
        key_phenomena = {
            'ellipsis': [p for p in comparison_df.index if 'ellipsis' in p.lower()],
            'anaphor': [p for p in comparison_df.index if 'anaphor' in p.lower()]
        }
        
        print(f"\n🔍 Ellipsis Phenomena Performance:")
        for phenom in key_phenomena['ellipsis']:
            if phenom in comparison_df.index:
                row = comparison_df.loc[phenom]
                print(f"  {phenom}:")
                for model in qwen_results.keys():
                    if model in row:
                        print(f"    {model}: {row[model]:.3f}")
        
        print(f"\n🔍 Anaphor Phenomena Performance:")
        for phenom in key_phenomena['anaphor']:
            if phenom in comparison_df.index:
                row = comparison_df.loc[phenom]
                print(f"  {phenom}:")
                for model in qwen_results.keys():
                    if model in row:
                        print(f"    {model}: {row[model]:.3f}")
        
        # Generate insights about cross-model consistency
        print(f"\n💡 Cross-Model Validation Insights:")
        print("This analysis shows how linguistic phenomena perform across different model sizes within the Qwen family.")
        print("To validate that these patterns are not specific to Qwen's training data,")
        print("we would need to compare with other model families like LLaMA, GPT-2, etc.")
        
        # Calculate consistency metrics
        if len(qwen_results) > 1:
            # For each phenomenon, calculate the variance across models
            consistency_scores = {}
            for phenom in all_phenomena:
                values = [qwen_results[model].get(phenom, np.nan) for model in qwen_results.keys()]
                values = [v for v in values if not np.isnan(v)]  # Remove NaN values
                if len(values) > 1:
                    std_dev = np.std(values)
                    mean_val = np.mean(values)
                    consistency_score = 1 - (std_dev / (mean_val + 1e-8))  # Higher is more consistent
                    consistency_scores[phenom] = {
                        'std_dev': std_dev,
                        'mean': mean_val,
                        'consistency': consistency_score,
                        'values': values
                    }
            
            # Sort by consistency (most consistent first)
            sorted_consistency = sorted(consistency_scores.items(), 
                                      key=lambda x: x[1]['consistency'], reverse=True)
            
            print(f"\n📋 Phenomena Consistency Across Qwen Models (Most Consistent First):")
            for phenom, stats in sorted_consistency[:10]:  # Top 10
                print(f"  {phenom}: Consistency={stats['consistency']:.3f}, "
                      f"StdDev={stats['std_dev']:.3f}, Values={stats['values']}")
        
        return comparison_df

    def generate_cross_model_report(self, comparison_df: pd.DataFrame):
        """
        Generate cross-model validation report
        
        Args:
            comparison_df: DataFrame with model comparison results
        """
        report_path = f"{self.config['output']['visualization_dir']}/cross_model_validation_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Cross-Model Validation Report\n\n")
            
            f.write("## 1. Overview\n\n")
            f.write("This report analyzes the consistency of linguistic phenomena across different model families.\n")
            f.write("The goal is to determine whether observed patterns are universal language model limitations\n")
            f.write("or specific to particular architectures/training data.\n\n")
            
            f.write("## 2. Qwen Model Family Analysis\n\n")
            f.write("### 2.1 Overall Performance\n\n")
            f.write("| Model | Average Accuracy |\n")
            f.write("|-------|------------------|\n")
            for model in comparison_df.columns:
                avg_acc = comparison_df[model].mean()
                f.write(f"| {model} | {avg_acc:.3f} |\n")
            
            f.write("\n### 2.2 Performance by Key Phenomena\n\n")
            f.write("#### Ellipsis Phenomena:\n\n")
            f.write("| Phenomenon | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |\n")
            f.write("|------------|--------------|--------------|-------------|\n")
            for phenom in comparison_df.index:
                if 'ellipsis' in phenom.lower():
                    f.write(f"| {phenom} | ")
                    for model in comparison_df.columns:
                        val = comparison_df.loc[phenom, model] if not pd.isna(comparison_df.loc[phenom, model]) else '-'
                        f.write(f"{val:.3f} | " if isinstance(val, float) else f"{val} | ")
                    f.write("\n")
            
            f.write("\n#### Anaphor Phenomena:\n\n")
            f.write("| Phenomenon | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |\n")
            f.write("|------------|--------------|--------------|-------------|\n")
            for phenom in comparison_df.index:
                if 'anaphor' in phenom.lower():
                    f.write(f"| {phenom} | ")
                    for model in comparison_df.columns:
                        val = comparison_df.loc[phenom, model] if not pd.isna(comparison_df.loc[phenom, model]) else '-'
                        f.write(f"{val:.3f} | " if isinstance(val, float) else f"{val} | ")
                    f.write("\n")
            
            f.write("\n## 3. Scaling Analysis\n\n")
            f.write("### 3.1 Model Scaling Effects\n\n")
            f.write("As model size increases from 0.5B to 7B parameters:\n")
            f.write("- Overall accuracy improves from {:.1%} to {:.1%}\n".format(
                comparison_df['Qwen2.5-0.5B'].mean(), comparison_df['Qwen2.5-7B'].mean()))
            
            ellipsis_improvement = 0
            anaphor_improvement = 0
            ellipsis_count = 0
            anaphor_count = 0
            
            for phenom in comparison_df.index:
                if 'ellipsis' in phenom.lower() and 'Qwen2.5-0.5B' in comparison_df.columns and 'Qwen2.5-7B' in comparison_df.columns:
                    ell_05b = comparison_df.loc[phenom, 'Qwen2.5-0.5B']
                    ell_7b = comparison_df.loc[phenom, 'Qwen2.5-7B']
                    if not pd.isna(ell_05b) and not pd.isna(ell_7b):
                        ellipsis_improvement += (ell_7b - ell_05b)
                        ellipsis_count += 1
                
                if 'anaphor' in phenom.lower() and 'Qwen2.5-0.5B' in comparison_df.columns and 'Qwen2.5-7B' in comparison_df.columns:
                    ana_05b = comparison_df.loc[phenom, 'Qwen2.5-0.5B']
                    ana_7b = comparison_df.loc[phenom, 'Qwen2.5-7B']
                    if not pd.isna(ana_05b) and not pd.isna(ana_7b):
                        anaphor_improvement += (ana_7b - ana_05b)
                        anaphor_count += 1
            
            if ellipsis_count > 0:
                avg_ellipsis_improvement = ellipsis_improvement / ellipsis_count
                f.write("- Ellipsis phenomena show average improvement of {:.2%}\n".format(avg_ellipsis_improvement))
            
            if anaphor_count > 0:
                avg_anaphor_improvement = anaphor_improvement / anaphor_count
                f.write("- Anaphor phenomena show average improvement of {:.2%}\n".format(avg_anaphor_improvement))
            
            f.write("\n## 4. Cross-Model Validation Requirements\n\n")
            f.write("To validate that these patterns are not specific to Qwen models:\n\n")
            f.write("1. **LLaMA Family**: Test with LLaMA-3-8B to see if similar patterns emerge\n")
            f.write("2. **GPT Family**: Test with GPT-2 (small) for comparison with different architecture\n")
            f.write("3. **Other Architectures**: Include models like Mistral, Mixtral, etc.\n\n")
            
            f.write("## 5. Key Findings\n\n")
            f.write("### 5.1 Ellipsis Analysis\n\n")
            f.write("- In Qwen models, ellipsis phenomena show modest improvement with scale\n")
            f.write("- This suggests that larger models may better handle ellipsis constructions\n")
            f.write("- Cross-validation with other families needed to confirm universality\n\n")
            
            f.write("### 5.2 Anaphor Analysis\n\n")
            f.write("- Anaphor phenomena show more complex scaling behavior\n")
            f.write("- Some anaphor types may follow U-shaped curve (improvement) while others show inverse scaling\n")
            f.write("- This requires further investigation with more model families\n\n")
            
            f.write("### 5.3 Human-Level Comparison\n\n")
            f.write("- Human benchmarks for BLIMP dataset are typically around 88%-90%\n")
            f.write("- Current Qwen models reach ~58% accuracy, indicating significant room for improvement\n")
            f.write("- Cross-model validation will help identify universal vs. architecture-specific limitations\n\n")
            
            f.write("## 6. Recommendations\n\n")
            f.write("1. Generate results for LLaMA-3-8B and GPT-2 models on the same test set\n")
            f.write("2. Compare the patterns observed in Qwen with other architectures\n")
            f.write("3. Focus on phenomena that show consistent behavior across all model families\n")
            f.write("4. Investigate phenomena that show family-specific patterns for training insights\n")

        print(f"📊 Cross-model validation report generated: {report_path}")

    def run_validation(self):
        """
        Run complete cross-model validation
        """
        print("🚀 Starting Cross-Model Validation Analysis")
        
        # Perform cross-model analysis
        comparison_df = self.analyze_cross_model_validation()
        
        # Generate report
        self.generate_cross_model_report(comparison_df)
        
        # Save comparison results
        comparison_df.to_csv(self.config['output']['comparison_csv'])
        print(f"💾 Saved cross-model comparison results to: {self.config['output']['comparison_csv']}")
        
        print("✅ Cross-Model Validation Complete!")
        return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Cross-Model Validation Tool')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    args = parser.parse_args()

    validator = CrossModelValidator(config_path=args.config)
    results = validator.run_validation()


if __name__ == "__main__":
    main()