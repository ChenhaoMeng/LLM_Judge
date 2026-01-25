import os
os.environ['HF_HOME'] = '/mnt/drive1/chenhao/cache/huggingface'

import torch
import pandas as pd
import json

import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 📁 路径辅助函数 =================
def get_project_root():
    """获取项目根目录（从core/目录向上两级）"""
    current_file = os.path.abspath(__file__)
    # 从 core/judge.py 向上到项目根
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()




MODEL_PATHS = {
    # --- Qwen 系列 (保持不变) ---
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-7b":   "Qwen/Qwen2.5-7B",
    "qwen-3b":   "Qwen/Qwen2.5-3B",


    "pythia-70m":  "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b":   "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b":  "EleutherAI/pythia-12b",
}
# ===========================================

class ModelScorer:
    def __init__(self, model_key, device="cuda" if torch.cuda.is_available() else "cpu", 
                 use_length_normalization=True, length_penalty_alpha=0.6):
        """
        Args:
            model_key: 模型标识符
            device: 计算设备
            use_length_normalization: 是否使用 SLLN-LP 长度归一化
            length_penalty_alpha: 长度惩罚系数，范围通常在 [0.5, 1.0]
                                 较小的值对短句更友好，较大的值对长句更公平
        """
        self.model_name = MODEL_PATHS.get(model_key, model_key)
        self.device = device
        self.model_key = model_key
        self.use_length_normalization = use_length_normalization
        self.length_penalty_alpha = length_penalty_alpha
        
        print(f"🚀 [Loading] 正在加载模型: {self.model_name} ...")
        if self.use_length_normalization:
            print(f"📏 [SLLN-LP] 启用长度归一化，惩罚系数 α = {self.length_penalty_alpha}")
        
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
        """
        计算句子评分，可选使用 SLLN-LP 长度归一化
        
        SLLN-LP (Sentence-Level Length Normalized - Length Penalty):
        normalized_score = raw_score / (length ^ alpha)
        
        这个公式特别适用于处理省略（Ellipsis）等会导致句子长度明显变化的语法现象
        """
        if not sentence:
            return -float('inf')
        
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.model.device)
        input_ids = inputs.input_ids
        seq_length = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            raw_score = -loss.item()  # 原始负对数似然分数
        
        # 应用 SLLN-LP 长度归一化
        if self.use_length_normalization and seq_length > 0:
            # 归一化公式: normalized_score = raw_score / (length ^ alpha)
            # 对于 loss，我们在 log 空间处理更稳定
            # score = -log(P) / (length ^ alpha) 等价于 score = -log(P) * (length ^ (-alpha))
            normalization_factor = seq_length ** (-self.length_penalty_alpha)
            normalized_score = raw_score * normalization_factor
            
            return normalized_score
        else:
            return raw_score

    def process_file(self, input_file, output_file):
        # ... (保持不变，已包含 pair_id 修复) ...
        print(f"📄 处理文件: {input_file}")
        try:
            df = pd.read_json(input_file, lines=True)
        except ValueError:
            print(f"⚠️ 文件格式错误或为空: {input_file}")
            return

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            s_good = row.get('sentence_good', '')
            s_bad = row.get('sentence_bad', '')
            score_good = self.get_sentence_score(s_good)
            score_bad = self.get_sentence_score(s_bad)
            is_correct = score_good > score_bad
            
            current_pair_id = row.get('pair_id')
            if pd.isna(current_pair_id) or current_pair_id is None:
                current_pair_id = idx

            # 计算长度信息（用于分析）
            len_good = len(self.tokenizer.encode(s_good, add_special_tokens=False))
            len_bad = len(self.tokenizer.encode(s_bad, add_special_tokens=False))

            result_row = {
                "UID": row.get('UID'),
                "file_source": row.get('file_source', os.path.basename(input_file)),
                "phenomenon": row.get('phenomenon'),
                "pair_id": current_pair_id,
                "sentence_good": s_good,
                "sentence_bad": s_bad,
                "score_good": score_good,
                "score_bad": score_bad,
                "len_good": len_good,
                "len_bad": len_bad,
                "is_correct": is_correct,
                "model": self.model_name,
                "use_length_norm": self.use_length_normalization,
                "length_penalty_alpha": self.length_penalty_alpha if self.use_length_normalization else None
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_row, ensure_ascii=False) + '\n')
        print(f"✅ 完成。保存至: {output_file}\n")

def run_batch_scoring():
    data_files = [
        os.path.join(PROJECT_ROOT, "data/anaphor_gender_agreement.jsonl"),
        os.path.join(PROJECT_ROOT, "data/anaphor_number_agreement.jsonl"),
        os.path.join(PROJECT_ROOT, "data/ellipsis_adj.jsonl"),
        os.path.join(PROJECT_ROOT, "data/ellipsis_double_object.jsonl"),
        os.path.join(PROJECT_ROOT, "data/ellipsis_n_bar_class.jsonl"),
        os.path.join(PROJECT_ROOT, "data/superlative_quantifiers_1.jsonl"),
        os.path.join(PROJECT_ROOT, "data/superlative_quantifiers_2.jsonl")
    ]
    
    target_models = [
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
    
    output_root = os.path.join(PROJECT_ROOT, "results_scoring")
    
    for model_key in target_models:
        try:
            model_dir = os.path.join(output_root, model_key)
            all_files_exist = True
            
            for filename in data_files:
                pure_filename = os.path.basename(filename)
                output_name = f"{os.path.splitext(pure_filename)[0]}_scored.jsonl"
                output_path = os.path.join(model_dir, output_name)
                if not os.path.exists(output_path):
                    all_files_exist = False
                    break
            
            if all_files_exist:
                print(f"⏭️  [跳过模型] {model_key} 的所有结果文件已存在。")
                continue 

            print(f"🚀 [Init] 准备加载模型: {model_key}")
            scorer = ModelScorer(model_key)
            
            os.makedirs(model_dir, exist_ok=True)
            
            for filename in data_files:
                if not os.path.exists(filename):
                    print(f"⚠️ [跳过] 找不到输入文件: {filename}")
                    continue
                
                pure_filename = os.path.basename(filename)
                output_name = f"{os.path.splitext(pure_filename)[0]}_scored.jsonl"
                output_path = os.path.join(model_dir, output_name)
                
                if os.path.exists(output_path):
                    print(f"⏭️  文件已存在，跳过: {output_path}")
                    continue
                    
                scorer.process_file(filename, output_path)
            
            del scorer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 模型 {model_key} 运行出错: {e}")
            torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    run_batch_scoring()