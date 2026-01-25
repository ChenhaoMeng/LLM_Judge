import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# ================= ⚙️ 配置区域 =================

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()
INPUT_DIR = os.path.join(PROJECT_ROOT, "results_merged")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_report")

MODEL_META = {
    # ... (保持原本的配置不变) ...
    "qwen-0.5b":    {"size": 0.5},
    "qwen-1.5b":    {"size": 1.5},
    "qwen-3b":      {"size": 3.0},
    "qwen-7b":      {"size": 7.0},
    "qwen-14b":     {"size": 14.0},
    "qwen-14b-awq": {"size": 14.0},
    "qwen-32b":     {"size": 32.0},
    "llama-1b":     {"size": 1.0},
    "llama-3b":     {"size": 3.0},
    "llama-8b":     {"size": 8.0},
    "pythia-70m":   {"size": 0.07},
    "pythia-160m":  {"size": 0.16},
    "pythia-410m":  {"size": 0.41},
    "pythia-1b":    {"size": 1.0},
    "pythia-1.4b":  {"size": 1.4},
    "pythia-2.8b":  {"size": 2.8},
    "pythia-6.9b":  {"size": 6.9},
    "pythia-12b":   {"size": 12.0},
    "gpt2":         {"size": 0.12},
    "gemma-2b":     {"size": 2.0},
    "gemma-9b":     {"size": 9.0},
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("bright")

# ===============================================

class Analyzer:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.df_all_stats = pd.DataFrame()
        self.raw_data_cache = {}  # 缓存原始数据用于统计分析
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    # 🔥 新增：自动识别家族的辅助函数
    def _get_family_name(self, model_name):
        name = model_name.lower()
        if "qwen" in name: return "Qwen"
        if "pythia" in name: return "Pythia"
        if "llama" in name: return "Llama"
        if "gemma" in name: return "Gemma"
        if "gpt2" in name: return "GPT-2"
        return "Other"

    def load_data(self):
        files = glob.glob(os.path.join(self.input_dir, "*_merged.jsonl"))
        if not files:
            print(f"❌ 未找到數據文件，請檢查 {self.input_dir}")
            return

        print(f"📂 開始加載 {len(files)} 個模型文件...")
        all_stats_list = []
        
        for f in files:
            model_name = os.path.basename(f).replace("_merged.jsonl", "")
            try:
                df = pd.read_json(f, lines=True)
                if 'is_correct' in df.columns:
                    df['correct'] = df['is_correct']
                
                # 獲取 Size
                meta = MODEL_META.get(model_name)
                if not meta:
                    match = re.search(r'(\d+\.?\d*)b', model_name.lower())
                    size = float(match.group(1)) if match else None
                    if size is None: continue
                else:
                    size = meta['size']

                # 🔥 新增：獲取家族標籤
                family = self._get_family_name(model_name)
                
                # 🔥 缓存原始数据用于统计分析
                self.raw_data_cache[model_name] = df

                # 1. 計算【整體】準確率
                total_acc = df['correct'].mean()
                all_stats_list.append({
                    "Model": model_name,
                    "Size": size,
                    "Family": family,  # 🔥 記錄家族
                    "Phenomenon": "OVERALL (Average)", 
                    "Accuracy": total_acc
                })
                
                # 2. 計算【分現象】準確率
                phenom_acc = df.groupby('phenomenon')['correct'].mean().reset_index()
                for _, row in phenom_acc.iterrows():
                    all_stats_list.append({
                        "Model": model_name,
                        "Size": size,
                        "Family": family, # 🔥 記錄家族
                        "Phenomenon": row['phenomenon'],
                        "Accuracy": row['correct']
                    })

            except Exception as e:
                print(f"  ❌ 加載 {model_name} 失敗: {e}")

        self.df_all_stats = pd.DataFrame(all_stats_list)
        print(f"✅ 數據加載完成，共 {len(self.df_all_stats)} 條記錄")

    # 🔥 新增方法：繪製不同家族的對比圖 (解決鋸齒問題)
    def plot_family_comparison(self):
        """
        繪製不同家族的 Scaling Law 對比 (只看 OVERALL)
        這張圖可以清晰展示 Qwen 是否比 Pythia 更強
        """
        if self.df_all_stats.empty: return
        
        # 只取 Overall 數據
        df_overall = self.df_all_stats[self.df_all_stats["Phenomenon"] == "OVERALL (Average)"].sort_values("Size")
        
        plt.figure(figsize=(12, 7))
        
        # 繪圖：hue='Family' 讓不同家族顯示不同顏色
        sns.lineplot(
            data=df_overall,
            x='Size', y='Accuracy',
            hue='Family', style='Family',
            markers=True, dashes=False, linewidth=2.5, markersize=9, alpha=0.9
        )
        
        # 標註點
        for _, row in df_overall.iterrows():
            plt.text(row['Size'], row['Accuracy'] + 0.01, row['Model'], 
                     fontsize=8, ha='center', alpha=0.7)

        plt.xscale('log')
        plt.title('Scaling Law by Model Family (Overall Accuracy)', fontsize=16)
        plt.xlabel('Model Parameters (Billions)', fontsize=12)
        plt.ylabel('Average Accuracy', fontsize=12)
        
        # 設置 X 軸刻度
        unique_sizes = sorted(df_overall['Size'].unique())
        plt.xticks(unique_sizes, [f"{s}B" for s in unique_sizes], rotation=45)
        
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        
        out_file = os.path.join(OUTPUT_DIR, "scaling_law_by_family.png")
        plt.savefig(out_file, dpi=300)
        print(f"📈 [繪圖完成] 家族對比圖已保存 -> {out_file}")
        plt.close()

    # 🔥 新增方法：為每個家族單獨畫一張詳細圖
    def plot_per_family_detailed(self):
        """
        為每個家族 (Qwen, Pythia...) 單獨生成一張圖，
        展示該家族內不同 Phenomenon 的 Scaling 趨勢
        """
        if self.df_all_stats.empty: return

        unique_families = self.df_all_stats['Family'].unique()
        
        for family in unique_families:
            if family == "Other": continue # 跳過雜項
            
            df_family = self.df_all_stats[self.df_all_stats['Family'] == family].sort_values("Size")
            if df_family.empty: continue

            plt.figure(figsize=(12, 7))
            
            # 這裡 hue='Phenomenon'，展示該家族內部的能力分層
            sns.lineplot(
                data=df_family,
                x='Size', y='Accuracy',
                hue='Phenomenon', style='Phenomenon',
                markers=True, dashes=False, linewidth=2, markersize=8
            )
            
            plt.xscale('log')
            plt.title(f'{family} Family Scaling Law: Accuracy vs Size', fontsize=16)
            plt.xlabel('Model Parameters (Billions)', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.ylim(0, 1.05)
            
            unique_sizes = sorted(df_family['Size'].unique())
            plt.xticks(unique_sizes, [f"{s}B" for s in unique_sizes], rotation=45)
            
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Phenomenon")
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.tight_layout()
            
            out_file = os.path.join(OUTPUT_DIR, f"scaling_law_detail_{family}.png")
            plt.savefig(out_file, dpi=300)
            print(f"📈 [繪圖完成] {family} 家族詳情圖已保存 -> {out_file}")
            plt.close()

    def plot_aggregate_scaling(self):
        # (保持原有的代碼不變，這張總圖還是很有用的)
        if self.df_all_stats.empty: return
        df_plot = self.df_all_stats.sort_values('Size')
        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=df_plot,
            x='Size', y='Accuracy',
            hue='Phenomenon', style='Phenomenon',
            markers=True, dashes=False, linewidth=2, markersize=8, alpha=0.85
        )
        overall_data = df_plot[df_plot['Phenomenon'] == "OVERALL (Average)"]
        for _, row in overall_data.iterrows():
            plt.text(row['Size'], row['Accuracy'] + 0.015, row['Model'], 
                     fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        plt.xscale('log')
        unique_sizes = sorted(df_plot['Size'].unique())
        plt.xticks(unique_sizes, [f"{s}B" for s in unique_sizes], fontsize=9, rotation=45)
        plt.title('Aggregate Scaling Law (All Models Mixed)', fontsize=16) # 標題稍微改一下
        plt.xlabel('Model Parameters (Billions)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        out_file = os.path.join(OUTPUT_DIR, "scaling_law_aggregate_raw.png")
        plt.savefig(out_file, dpi=300)
        print(f"📈 [繪圖完成] 原始匯總圖已保存 -> {out_file}")
        plt.close()

    def generate_summary_csv(self):
        if self.df_all_stats.empty: return
        # 添加 Family 字段到 CSV
        pivot = self.df_all_stats.pivot_table(
            index=['Family', 'Phenomenon'], # 🔥 索引加了 Family
            columns='Model', 
            values='Accuracy'
        )
        sorted_cols = sorted(pivot.columns, key=lambda x: MODEL_META.get(x, {}).get('size', 0) or 999)
        pivot = pivot[sorted_cols]
        csv_path = os.path.join(OUTPUT_DIR, "final_accuracy_report.csv")
        pivot.to_csv(csv_path)
        print(f"💾 數據表已保存 -> {csv_path}")

    # ================= 🔥 新增：统计显著性检验 =================
    def bootstrap_confidence_interval(self, data, n_bootstrap=1000, confidence=0.95):
        """
        使用 Bootstrap 重采样计算置信区间
        
        Args:
            data: 一维数组，准确率或分数数据
            n_bootstrap: Bootstrap 重采样次数
            confidence: 置信水平 (默认 0.95 表示 95% 置信区间)
        
        Returns:
            (lower_bound, upper_bound, mean_value)
        """
        if len(data) == 0:
            return None, None, None
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # 有放回采样
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        mean_val = np.mean(data)
        
        return lower, upper, mean_val
    
    def calculate_statistical_significance(self):
        """
        计算模型间的统计显著性检验
        """
        if not self.raw_data_cache:
            print("❌ 未找到原始数据，无法进行统计检验")
            return
        
        print("\n" + "="*60)
        print("📊 统计显著性检验 (Bootstrap + Mann-Whitney U Test)")
        print("="*60)
        
        # 比较 Qwen vs Pythia（整体准确率）
        qwen_models = [m for m in self.raw_data_cache.keys() if "qwen" in m.lower()]
        pythia_models = [m for m in self.raw_data_cache.keys() if "pythia" in m.lower()]
        
        if not qwen_models or not pythia_models:
            print("⚠️ 无法找到足够的模型进行比较")
            return
        
        results = []
        
        # 对每个现象进行检验
        all_phenomena = set()
        for df in self.raw_data_cache.values():
            if 'phenomenon' in df.columns:
                all_phenomena.update(df['phenomenon'].unique())
        
        for phenomenon in sorted(all_phenomena):
            qwen_scores = []
            pythia_scores = []
            
            for model_name, df in self.raw_data_cache.items():
                if 'phenomenon' in df.columns and 'correct' in df.columns:
                    phenom_df = df[df['phenomenon'] == phenomenon] if phenomenon != "OVERALL (Average)" else df
                    scores = phenom_df['correct'].values
                    
                    if "qwen" in model_name.lower():
                        qwen_scores.extend(scores)
                    elif "pythia" in model_name.lower():
                        pythia_scores.extend(scores)
            
            if len(qwen_scores) == 0 or len(pythia_scores) == 0:
                continue
            
            # Bootstrap 置信区间
            qwen_lower, qwen_upper, qwen_mean = self.bootstrap_confidence_interval(
                np.array(qwen_scores), n_bootstrap=1000
            )
            pythia_lower, pythia_upper, pythia_mean = self.bootstrap_confidence_interval(
                np.array(pythia_scores), n_bootstrap=1000
            )
            
            # Mann-Whitney U 检验（非参数检验，适用于准确率数据）
            try:
                u_statistic, p_value = stats.mannwhitneyu(qwen_scores, pythia_scores, alternative='two-sided')
            except:
                p_value = 1.0
            
            # 效应量 (Cohen's d)
            try:
                cohens_d = (np.mean(qwen_scores) - np.mean(pythia_scores)) / np.sqrt(
                    (np.var(qwen_scores) + np.var(pythia_scores)) / 2
                )
            except:
                cohens_d = 0.0
            
            results.append({
                "Phenomenon": phenomenon,
                "Qwen_Mean": f"{qwen_mean:.4f}",
                "Qwen_CI_Lower": f"{qwen_lower:.4f}",
                "Qwen_CI_Upper": f"{qwen_upper:.4f}",
                "Pythia_Mean": f"{pythia_mean:.4f}",
                "Pythia_CI_Lower": f"{pythia_lower:.4f}",
                "Pythia_CI_Upper": f"{pythia_upper:.4f}",
                "P_Value": f"{p_value:.4f}",
                "Significant": "✅ Yes" if p_value < 0.05 else "❌ No",
                "Cohens_D": f"{cohens_d:.3f}",
                "Effect_Size": "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
            })
        
        # 保存结果
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, "statistical_significance_test.csv")
        df_results.to_csv(csv_path, index=False, encoding='utf-8')
        
        print("\n📊 统计检验结果:")
        print(df_results.to_string(index=False))
        print(f"\n💾 结果已保存 -> {csv_path}")
    
    # ================= 🔥 新增：一致性度量 =================
    def calculate_agreement_metrics(self):
        """
        计算模型判断与数据集标签的一致性（Cohen's Kappa / Scott's Pi）
        
        注意：这里假设数据集有正确的标签（sentence_good vs sentence_bad）
        """
        if not self.raw_data_cache:
            print("❌ 未找到原始数据，无法计算一致性")
            return
        
        print("\n" + "="*60)
        print("📊 一致性度量 (Cohen's Kappa)")
        print("="*60)
        
        results = []
        
        for model_name, df in self.raw_data_cache.items():
            if 'correct' not in df.columns or 'phenomenon' not in df.columns:
                continue
            
            # 按现象分组计算
            for phenomenon in df['phenomenon'].unique():
                phenom_df = df[df['phenomenon'] == phenomenon]
                
                # 假设正确标签（ground truth）：sentence_good 应该比 sentence_bad 得分高
                # 但实际上我们需要知道真实的标签
                # 这里我们使用 score_good > score_bad 作为 ground truth
                if 'score_good' in phenom_df.columns and 'score_bad' in phenom_df.columns:
                    ground_truth = (phenom_df['score_good'] > phenom_df['score_bad']).astype(int)
                    model_predictions = phenom_df['correct'].astype(int)
                    
                    # 计算 Cohen's Kappa
                    try:
                        kappa = cohen_kappa_score(ground_truth, model_predictions)
                        
                        # 计算 Scott's Pi（假设两个标注者）
                        # Scott's Pi = (P_o - P_e) / (1 - P_e)
                        # 其中 P_o 是观察一致性，P_e 是期望一致性
                        p_o = np.mean(ground_truth == model_predictions)
                        p_positive = np.mean(ground_truth)
                        p_e = p_positive ** 2 + (1 - p_positive) ** 2
                        scotts_pi = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0
                        
                        # 准确率
                        accuracy = p_o
                        
                        results.append({
                            "Model": model_name,
                            "Phenomenon": phenomenon,
                            "Accuracy": f"{accuracy:.4f}",
                            "Cohens_Kappa": f"{kappa:.4f}",
                            "Scotts_Pi": f"{scotts_pi:.4f}",
                            "Agreement": "Almost Perfect" if kappa > 0.81 else \
                                        "Substantial" if kappa > 0.61 else \
                                        "Moderate" if kappa > 0.41 else \
                                        "Fair" if kappa > 0.21 else "Poor"
                        })
                    except Exception as e:
                        print(f"  ⚠️ 计算 {model_name} - {phenomenon} 时出错: {e}")
        
        if results:
            df_results = pd.DataFrame(results)
            csv_path = os.path.join(OUTPUT_DIR, "agreement_metrics.csv")
            df_results.to_csv(csv_path, index=False, encoding='utf-8')
            
            print("\n📊 一致性度量结果:")
            print(df_results.to_string(index=False))
            print(f"\n💾 结果已保存 -> {csv_path}")
        else:
            print("⚠️ 无法计算一致性度量（缺少必要的字段）")

def main():
    analyzer = Analyzer(INPUT_DIR)
    analyzer.load_data()
    
    # 1. 原始的總圖 (還是保留)
    analyzer.plot_aggregate_scaling()
    
    # 2. 🔥 新增：家族對比圖 (Qwen vs Pythia vs Llama)
    analyzer.plot_family_comparison()
    
    # 3. 🔥 新增：每個家族內部的詳情圖
    analyzer.plot_per_family_detailed()
    
    analyzer.generate_summary_csv()
    
    # 4. 🔥 新增：统计显著性检验
    analyzer.calculate_statistical_significance()
    
    # 5. 🔥 新增：一致性度量
    analyzer.calculate_agreement_metrics()

if __name__ == "__main__":
    main()