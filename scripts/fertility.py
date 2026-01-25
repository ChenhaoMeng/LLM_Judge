import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 获取项目根目录
def get_project_root():
    """获取项目根目录（从scripts/目录向上）"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

# 1. 输入数据目录 (读取该目录下所有 .jsonl 文件)
INPUT_DIR = os.path.join(PROJECT_ROOT, "data")

# 2. 输出结果路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_report")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "token_fertility_all.csv")
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "token_fertility_comparison.png")

# 3. 模型列表 (显示名称 : HuggingFace ID)
MODEL_MAP = {
    # --- Native Chinese Models (Treatment Group) ---
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-3B":   "Qwen/Qwen2.5-3B",
    "Qwen2.5-7B":   "Qwen/Qwen2.5-7B",
    
    # --- English-Centric Models (Control Group) ---
    "Pythia-1.4B":  "EleutherAI/pythia-1.4b",
    "Pythia-2.8B":  "EleutherAI/pythia-2.8b",
    "Pythia-6.9B":  "EleutherAI/pythia-6.9b",
    
    # --- Legacy Baseline ---
    "GPT-2":        "gpt2"
}

# 基准模型（用于计算 Ratio，通常选效果最好的中文模型）
BASELINE_MODEL = "Qwen2.5-7B"

# ================= 🚀 核心逻辑 =================

def load_all_sentences(input_dir):
    """读取目录下所有 jsonl 文件的 sentence_good"""
    all_sentences = []
    files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    print(f"📂 正在扫描目录: {input_dir}")
    print(f"   发现 {len(files)} 个文件。")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # 跳过非数据文件（如 output 结果）
        if "result" in filename or "test" in filename or "train" in filename:
            continue
            
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 取 sentence_good 作为标准中文样本
                    if 'sentence_good' in item:
                        all_sentences.append(item['sentence_good'])
                        count += 1
                except:
                    continue
        # print(f"   - {filename}: 加载 {count} 条")
    
    print(f"✅ 总计加载句子: {len(all_sentences)} 条")
    return all_sentences

def compute_fertility():
    sentences = load_all_sentences(INPUT_DIR)
    if not sentences:
        print("❌ 未找到数据，请检查路径。")
        return

    results = []
    baseline_avg = None

    print("\n🚀 开始计算 Token Fertility (词元生育率)...")
    
    # 这里我们按 MODEL_MAP 的顺序处理
    # 注意：同一家族（如Qwen系列）通常共用一个Tokenizer，可以缓存以加速，
    # 但为了严谨起见，我们还是逐个加载（因为不同大小的模型配置可能微调）。
    
    # 简单的缓存机制，避免重复加载完全相同的 tokenizer
    loaded_tokenizers = {} 

    for model_name, model_path in MODEL_MAP.items():
        print(f"🔹 [{model_name}] Loading Tokenizer...", end="")
        
        try:
            # 尝试从缓存获取 (Qwen系列大概率是一样的)
            if model_path in loaded_tokenizers:
                tokenizer = loaded_tokenizers[model_path]
                print(" (Cached)", end="")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                # 简单的优化：同一家族通常路径前缀一样，这里不做复杂判断，直接跑
            
            # 计算总 token 数
            total_tokens = 0
            # 批量 encode 会比逐条快，但为了进度条显示，我们用列表推导
            # tokenizer.encode 通常很快
            batch_tokens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
            total_tokens = sum(batch_tokens)
            
            avg_len = total_tokens / len(sentences)
            
            results.append({
                "Model": model_name,
                "Model_Path": model_path,
                "Avg_Tokens": avg_len
            })
            
            print(f" -> Avg: {avg_len:.2f}")

        except Exception as e:
            print(f"\n❌ Error loading {model_name}: {e}")

    # --- 数据分析 ---
    df = pd.DataFrame(results)
    
    # 获取基准模型的数值
    try:
        base_row = df[df['Model'] == BASELINE_MODEL]
        if not base_row.empty:
            baseline_avg = base_row['Avg_Tokens'].values[0]
        else:
            # 如果没找到指定的基准，就用列表里的最小值（通常是Qwen）
            baseline_avg = df['Avg_Tokens'].min()
    except:
        baseline_avg = df['Avg_Tokens'].min()

    # 计算 Ratio
    df['Fertility_Ratio'] = df['Avg_Tokens'] / baseline_avg
    df['Fertility_Ratio'] = df['Fertility_Ratio'].round(2)
    df['Avg_Tokens'] = df['Avg_Tokens'].round(2)

    # 排序：按 Ratio 升序 (越小越好)
    df = df.sort_values("Fertility_Ratio")

    # 打印报告
    print("\n" + "="*60)
    print(f"📊 TOKEN FERTILITY REPORT (Baseline: {BASELINE_MODEL})")
    print("="*60)
    print(df[['Model', 'Avg_Tokens', 'Fertility_Ratio']].to_string(index=False))
    print("="*60)
    
    # 保存
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 数据已保存 -> {OUTPUT_CSV}")
    
    return df

def plot_results(df):
    if df is None or df.empty: return

    # 设置风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 颜色映射：区分 Qwen, Pythia, GPT2
    def get_color(name):
        if "Qwen" in name: return "#1f77b4" # Blue
        if "Pythia" in name: return "#ff7f0e" # Orange
        return "#7f7f7f" # Gray

    colors = [get_color(name) for name in df['Model']]

    # 绘图
    ax = sns.barplot(x="Model", y="Avg_Tokens", data=df, palette=colors)
    
    # 添加数值标签
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ratio = df.iloc[i]['Fertility_Ratio']
        ax.text(p.get_x() + p.get_width()/2., height + 0.5, 
                f'{height:.1f}\n(x{ratio})', 
                ha="center", va="bottom", fontsize=10, color='black')

    plt.title("Token Fertility Analysis: Average Tokens per Chinese Sentence", fontsize=16)
    plt.ylabel("Average Token Count (Lower is Better)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, df['Avg_Tokens'].max() * 1.15) # 留出顶部空间
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"📈 图表已保存 -> {OUTPUT_IMG}")

if __name__ == "__main__":
    df = compute_fertility()
    plot_results(df)