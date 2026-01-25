import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= ⚙️ 1. 配置與路徑 =================
# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

# 確保 BASE_MODEL 與訓練時一致
BASE_MODEL = "Qwen/Qwen2.5-3b-instruct" 
# 訓練輸出的 LoRA 權重路徑（相對於項目根目錄）
LORA_PATH = os.path.join(PROJECT_ROOT, "lora_results", "Qwen2.5-3b-instruct")
# 測試數據集路徑（使用相對路徑，自動適配）
# 🔥 优先使用修复后的数据文件
TEST_FILE = os.path.join(PROJECT_ROOT, "finetune_multitask_test_fixed.jsonl")
# 如果不存在，尝试其他文件名
if not os.path.exists(TEST_FILE):
    TEST_FILE = os.path.join(PROJECT_ROOT, "finetune_multitask_test.jsonl")
if not os.path.exists(TEST_FILE):
    TEST_FILE = os.path.join(PROJECT_ROOT, "finetune_test.jsonl")

# 結果輸出路徑（使用fixed后缀区分）
OUTPUT_RESULT = os.path.join(PROJECT_ROOT, "lora_test_results", "test_results_3b_fixed.json")


BATCH_SIZE = 8 

# 🔧 環境變量配置（可選，根據實際需求調整）
# 如果不需要指定 HuggingFace 緩存目錄，可以註釋掉這行
# os.environ['HF_HOME'] = '/mnt/drive1/chenhao/cache/huggingface'


# ================= 🛠️ 2. 模型載入邏輯 =================
def load_inference_model():
    print(f"🚀 正在加載基礎模型: {BASE_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, 
        trust_remote_code=False,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    attn_impl = "sdpa"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation=attn_impl
    )
    

    print(f"📏 調整模型詞表大小以匹配 Tokenizer ({len(tokenizer)})...")
    base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"🔗 正在加載 LoRA 權重: {LORA_PATH}")
    try:
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
    except Exception as e:
        print(f"❌ 加載 LoRA 失敗，請檢查 LORA_PATH 是否正確。錯誤資訊: {e}")
        return None, None
        
    model.config.use_cache = True 
    model.eval()
    return model, tokenizer

# ================= 🧪 3. 批量測試邏輯 =================
def run_batch_test(model, tokenizer):
    if model is None: return
    
    results = []
    
    if not os.path.exists(TEST_FILE):
        print(f"❌ 找不到測試文件: {TEST_FILE}")
        print(f"   請先運行 python core/finetune_dataset.py 生成修復後的數據")
        return

    output_dir = os.path.dirname(OUTPUT_RESULT)
    if output_dir and not os.path.exists(output_dir):
        print(f"📁 正在創建目錄: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    # 🔥 顯示數據信息
    is_fixed_data = "fixed" in TEST_FILE
    print(f"\n📊 測試數據信息:")
    print(f"   - 文件路徑: {TEST_FILE}")
    print(f"   - 數據類型: {'修復後的數據（無泄露）' if is_fixed_data else '原始數據'}")
    print(f"   - 樣本數量: {len(test_data)}")
    
    # 統計任務類型
    task_counts = {"MC": 0, "Judge": 0, "Correction": 0}
    for item in test_data:
        inst = item.get("instruction", "")
        if "選出" in inst or "選項" in inst:
            task_counts["MC"] += 1
        elif "判斷" in inst:
            task_counts["Judge"] += 1
        elif "修改" in inst or "修正" in inst:
            task_counts["Correction"] += 1
    
    print(f"   - 任務分布: MC={task_counts['MC']}, Judge={task_counts['Judge']}, Correction={task_counts['Correction']}")

    dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n🧪 開始批量測試 (總計 {len(test_data)} 條，Batch Size: {BATCH_SIZE})...")
    
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            instructions = batch.get("instruction", [])
            user_inputs = batch.get("input", [])
            expected_outputs = batch.get("output", [])
            
            prompts = [f"User: {inst}\n{inp}\nAssistant: " for inst, inp in zip(instructions, user_inputs)]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            input_length = inputs.input_ids.shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated_ids = outputs[:, input_length:]
            batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 整理結果
            for i in range(len(batch_responses)):
                results.append({
                    "instruction": instructions[i],
                    "input": user_inputs[i],
                    "expected_output": expected_outputs[i],
                    "model_output": batch_responses[i].strip()
                })


    with open(OUTPUT_RESULT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 測試完成！結果已保存至: {OUTPUT_RESULT}")

# ================= 🏁 4. 啟動入口 =================
if __name__ == "__main__":

    torch.cuda.empty_cache()
    

    model, tokenizer = load_inference_model()

    run_batch_test(model, tokenizer)