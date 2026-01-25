import json
import os
import glob
import re
import pandas as pd
import nltk
from nltk.translate.gleu_score import sentence_gleu

# ================= ⚙️ 配置区域 =================

# 获取项目根目录
def get_project_root():
    """获取项目根目录（从scripts/目录向上）"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

# 你的结果文件所在目录
INPUT_DIR = os.path.join(PROJECT_ROOT, "lora_test_results") 
# 输出报表目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_report")

# 语法关键词映射
GRAMMAR_KEYWORDS = {
    "anaphor": ["代词", "先行词", "指代"],
    "ellipsis": ["省略"],
    "quantifiers": ["量词", "逻辑"]
}

def clean_text(text):
    """清洗模型输出，去除 Prompt 回声，提取纯净文本"""
    if not isinstance(text, str): return ""
    # 去除常见前缀
    text = re.sub(r"(修正后的句子[：:]\s*)", "", text)
    text = re.sub(r"(修改后的句子[：:]\s*)", "", text)
    text = re.sub(r"(正确的句子[：:]\s*)", "", text)
    text = re.sub(r"(答案[：:]\s*)", "", text)
    # 去除首尾空白
    return text.strip()

def calculate_gleu(reference, hypothesis):
    """计算单句 GLEU (按字分词)"""
    ref_tokens = [list(reference)] 
    hyp_tokens = list(hypothesis)
    if not hyp_tokens: hyp_tokens = [""] # 防止空串报错
    try:
        return sentence_gleu(ref_tokens, hyp_tokens)
    except:
        return 0.0

# ================= 🚀 核心逻辑 =================

def process_single_file(file_path):
    filename = os.path.basename(file_path)
    model_name = filename.replace("test_results_", "").replace(".json", "")
    
    print(f"📄 正在分析模型: {model_name} ...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    metrics = {
        "mc": {"correct": 0, "total": 0},
        "judge": {"correct": 0, "total": 0},
        "correction": {"gleu_sum": 0.0, "em_correct": 0, "total": 0}
    }
    
    grammar_breakdown = {k: {"gleu_sum": 0.0, "total": 0} for k in GRAMMAR_KEYWORDS}

    for item in data:
        instruction = item.get("instruction", "")
        expected = clean_text(item.get("expected_output", ""))
        model_out = clean_text(item.get("model_output", ""))

        # 1. 判定任务类型
        if "选出" in instruction or "选项" in instruction:
            mode = "mc"
        elif "判断" in instruction:
            mode = "judge"
        elif "修改" in instruction or "修正" in instruction:
            mode = "correction"
        else:
            continue # 跳过不明任务

        # 2. 判定语法类型
        g_type = "other"
        for k, v in GRAMMAR_KEYWORDS.items():
            if any(kw in instruction for kw in v):
                g_type = k
                break
        

        if mode == "mc":
            metrics["mc"]["total"] += 1

            target = expected.split(" ")[0] if " " in expected else expected
            if target in model_out:
                metrics["mc"]["correct"] += 1


        elif mode == "judge":
            metrics["judge"]["total"] += 1
  
            if model_out and expected and model_out[0] == expected[0]:
                metrics["judge"]["correct"] += 1

        elif mode == "correction":
            metrics["correction"]["total"] += 1
            

            if model_out == expected:
                metrics["correction"]["em_correct"] += 1

            score = calculate_gleu(expected, model_out)
            metrics["correction"]["gleu_sum"] += score

            if g_type in grammar_breakdown:
                grammar_breakdown[g_type]["total"] += 1
                grammar_breakdown[g_type]["gleu_sum"] += score


    mc_acc = metrics["mc"]["correct"] / metrics["mc"]["total"] if metrics["mc"]["total"] > 0 else 0
    judge_acc = metrics["judge"]["correct"] / metrics["judge"]["total"] if metrics["judge"]["total"] > 0 else 0
    
    corr_total = metrics["correction"]["total"]
    corr_gleu = metrics["correction"]["gleu_sum"] / corr_total if corr_total > 0 else 0
    corr_em = metrics["correction"]["em_correct"] / corr_total if corr_total > 0 else 0

    row = {
        "Model": model_name,
        "MC_Acc": mc_acc,
        "Judge_Acc": judge_acc,
        "Correction_GLEU": corr_gleu, 
        "Correction_EM": corr_em, 
        "Total_Samples": len(data)
    }


    for g_key, val in grammar_breakdown.items():
        avg_g_gleu = val["gleu_sum"] / val["total"] if val["total"] > 0 else 0
        row[f"GLEU_{g_key.capitalize()}"] = avg_g_gleu

    return row

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录不存在: {INPUT_DIR}")
        return

    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not files:
        print(f"❌ 目录下没有 JSON 文件")
        return

    all_results = []
    print(f"🚀 开始批量评估 {len(files)} 个文件...\n")

    for f in files:
        try:
            res = process_single_file(f)
            all_results.append(res)
        except Exception as e:
            print(f"❌ 处理 {f} 失败: {e}")


    df = pd.DataFrame(all_results)

    def extract_size(name):
        match = re.search(r"(\d+\.?\d*)b", name.lower())
        return float(match.group(1)) if match else 0
    
    df["Param_Size"] = df["Model"].apply(extract_size)
    df = df.sort_values("Param_Size") # 按大小排序
    df = df.drop(columns=["Param_Size"]) # 移除辅助列

    # 打印简报
    print("\n" + "="*60)
    print("🏆 FINAL EVALUATION LEADERBOARD (Sorted by Size)")
    print("="*60)
    # 只打印关键列
    print(df[["Model", "MC_Acc", "Correction_GLEU", "GLEU_Quantifiers", "GLEU_Anaphor"]].to_string(index=False, float_format="%.4f"))
    print("="*60)

    # 保存
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    out_path = os.path.join(OUTPUT_DIR, "final_leaderboard.csv")
    df.to_csv(out_path, index=False)
    print(f"\n💾 完整数据已保存至: {out_path}")
    print("   (包含 MC, Judge, Correction, 以及各项语法的细分 GLEU)")

if __name__ == "__main__":
    main()