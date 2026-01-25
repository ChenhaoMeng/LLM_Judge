import pandas as pd
import os
import glob
from tqdm import tqdm

# ================= 配置区域 =================

# 获取项目根目录
def get_project_root():
    """获取项目根目录（从scripts/目录向上）"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

INPUT_ROOT_DIR = os.path.join(PROJECT_ROOT, "results_scoring")

# 结果保存目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_merged")

# 你想合并的模型列表 (如果留空 []，则自动扫描 results_scoring 下的所有文件夹)
TARGET_MODELS = [
    "qwen-0.5b", 
    "qwen-1.5b", 
    "qwen-7b",
    "qwen-3b",
    "qwen-14b",
    "gpt2",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b"
] 

# ===========================================

def merge_model_results():
    # 1. 确定要处理的模型文件夹
    if not TARGET_MODELS:
        # 自动扫描目录下的所有文件夹
        model_dirs = [d for d in os.listdir(INPUT_ROOT_DIR) if os.path.isdir(os.path.join(INPUT_ROOT_DIR, d))]
    else:
        model_dirs = TARGET_MODELS

    if not model_dirs:
        print(f"❌ 在 {INPUT_ROOT_DIR} 下未找到任何模型文件夹！")
        return

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 将合并以下模型的数据: {model_dirs}\n")

    for model_name in model_dirs:
        model_path = os.path.join(INPUT_ROOT_DIR, model_name)
        
        # 2. 获取该模型下所有的 .jsonl 文件
        jsonl_files = glob.glob(os.path.join(model_path, "*.jsonl"))
        
        if not jsonl_files:
            print(f"⚠️  [跳过] {model_name} 目录下没有找到 .jsonl 文件")
            continue

        # 🔥【关键步骤】🔥：必须按文件名排序！
        # 只有这样，不同模型合并后的顺序才能保证一致（例如都先读 anaphor，再读 ellipsis）
        jsonl_files.sort()
        
        print(f"🚀 正在合并 {model_name} (共 {len(jsonl_files)} 个文件)...")
        
        all_data = []
        
        # 3. 逐个读取并合并
        for file_path in tqdm(jsonl_files, desc=f"Reading {model_name}", unit="file"):
            try:
                df = pd.read_json(file_path, lines=True)
                # 确保 file_source 列存在，方便后续知道这行数据来自哪个任务
                if 'file_source' not in df.columns:
                    df['file_source'] = os.path.basename(file_path)
                all_data.append(df)
            except ValueError:
                print(f"  ⚠️ 读取失败或文件为空: {file_path}")
        
        if not all_data:
            continue

        # 4. 合并成一个大 DataFrame
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 5. 【核心需求】重写 pair_id 为行号
        # 无论原始 pair_id 是什么，现在统一用 0, 1, 2, 3... 覆盖
        # 这样不同模型之间只要文件读取顺序一致，ID 就能对齐
        merged_df['pair_id'] = range(len(merged_df))
        
        # 6. 保存合并后的文件
        output_filename = f"{model_name}_merged.jsonl"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        merged_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        
        print(f"✅ {model_name} 合并完成！")
        print(f"   📊 总样本数: {len(merged_df)}")
        print(f"   💾 保存至: {output_path}")
        print("-" * 50)

if __name__ == "__main__":
    merge_model_results()