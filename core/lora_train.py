import os
import sys

# ================= 🔧 1. 环境与缓存配置 (必须在最前面) =================
os.environ['HF_HOME'] = '/mnt/drive1/chenhao/cache/huggingface'

os.environ['DISABLE_FLASH_ATTN'] = '1'

import torch
import gc
import time
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# ================= ⚙️ 2. 实验参数配置 =================

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

# 🔥 優先使用修復後的數據（無泄露）
USE_FIXED_DATA = True  # 設為False可使用原始數據

if USE_FIXED_DATA:
    DATA_FILES = {
        "train": os.path.join(PROJECT_ROOT, "finetune_multitask_train_fixed.jsonl"),
        "test": os.path.join(PROJECT_ROOT, "finetune_multitask_test_fixed.jsonl")
    }
else:
    DATA_FILES = {
        "train": os.path.join(PROJECT_ROOT, "finetune_train.jsonl"),
        "test": os.path.join(PROJECT_ROOT, "finetune_test.jsonl")
    }

MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct", 
]

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "lora_results")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "scaling_law_training_log.csv")

# ================= 🧹 3. 显存清理工具 =================
def clear_gpu_memory():
    """强制清理显存，防止 OOM"""
    print("🧹 Cleaning up GPU memory...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    except Exception as e:
        print(f"   Warning during cleanup: {e}")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU 0 - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# ================= 🚀 4. 单模型训练核心函数 =================
def train_one_model(model_name):
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for: {model_name}")
    print(f"{'='*60}")

    short_name = model_name.split("/")[-1]
    output_dir = os.path.join(OUTPUT_ROOT, short_name)
    
    # 清理之前的显存
    clear_gpu_memory()

    # --- A. 智能加载策略 ---
    is_large_model = "14B" in model_name or "32B" in model_name or "72B" in model_name
    
    if is_large_model:
        print("⚡ Large model detected: Enabling 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_dtype = None 
    else:
        print("🚀 Small/Medium model detected: Using Native FP16...")
        bnb_config = None
        model_dtype = torch.float16

    # --- B. 加载模型 ---
    print(f"📥 Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=model_dtype, 
        device_map="auto" if is_large_model else None, 
        trust_remote_code=True,
        attn_implementation="eager",  
    )
    
    # 小模型手动移到GPU
    if not is_large_model:
        model = model.to("cuda:0")
    
    # 为量化模型准备训练
    if is_large_model:
        model = prepare_model_for_kbit_training(model)
    
    # 禁用缓存（节省显存）
    model.config.use_cache = False

    # --- C. 加载 Tokenizer ---
    print(f"📥 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        print(f"⚠️ Fast tokenizer failed ({e}), using slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            use_fast=False
        )

    # 设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # --- D. LoRA 配置 ---
    print("🔧 Applying LoRA configuration...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 明确指定模块
    )
    
    # 应用 LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 🔥 确保模型在单一设备上
    if not is_large_model:
        print("📌 Ensuring model is on single device (cuda:0)...")
        model = model.to("cuda:0")

    # --- E. 数据预处理 ---
    print("📂 Loading and processing dataset...")
    try:
        raw_datasets = load_dataset("json", data_files=DATA_FILES)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        raise

    def format_and_tokenize(example):
        """格式化并tokenize样本"""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # 构建 prompt
        if input_text:
            text = f"User: {instruction}\n{input_text}\nAssistant: {output}"
        else:
            text = f"User: {instruction}\nAssistant: {output}"
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None,
        )
        
        # 将 input_ids 同时作为 labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    # 处理数据集
    dataset = raw_datasets.map(
        format_and_tokenize,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing dataset",
        batched=False,
    )
    
    print(f"✅ Dataset ready:")
    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Test samples: {len(dataset['test'])}")

    # --- F. 训练参数 (使用 TrainingArguments) ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2 if is_large_model else 4,  # 大模型用更小的batch
        per_device_eval_batch_size=2 if is_large_model else 4,
        gradient_accumulation_steps=8 if is_large_model else 4,  # 保持有效batch size
        learning_rate=2e-4,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        bf16=False,  # 明确禁用 bf16
        optim="paged_adamw_32bit",
        gradient_checkpointing=False,  # 🔥 禁用以避免SegFault
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,  # 🔥 禁用多进程加载以提高稳定性
        dataloader_pin_memory=True,
    )

    # --- G. Trainer 初始化 ---
    print("🏗️ Initializing SFTTrainer...")
    
    # 数据集已经预处理好，直接使用最简参数
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # --- H. 开始训练 ---
    print("🔥 Training started...")
    print(f"   Total steps: {len(dataset['train']) * 3 // (4 * 4)}")  # epochs * samples / (batch_size * grad_accum)
    
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        final_loss = train_result.training_loss
        print(f"✅ Training completed! Final loss: {final_loss:.4f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️ Training duration: {duration/60:.2f} minutes")

    # --- I. 保存模型 ---
    print(f"💾 Saving model to {output_dir}")
    try:
        # 保存最终模型
        final_output_dir = os.path.join(output_dir, "final_model")
        trainer.model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print(f"✅ Model saved successfully to {final_output_dir}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save model: {e}")

    # --- J. 清理 ---
    print("🧹 Cleaning up...")
    del model
    del tokenizer
    del trainer
    del dataset
    clear_gpu_memory()

    return final_loss, duration

# ================= 🔄 5. 主循环逻辑 =================
def main():
    print("="*60)
    print("🎯 Scaling Law LoRA Training Pipeline")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 检查数据文件
    for split, filepath in DATA_FILES.items():
        if not os.path.exists(filepath):
            print(f"❌ Error: {split} file not found: {filepath}")
            return
        print(f"✅ Found {split} file: {filepath}")
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("❌ Error: No GPU available!")
        return
    
    print(f"\n🎮 GPU Info:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"\n📋 Training Queue: {len(MODEL_LIST)} models")
    for i, model in enumerate(MODEL_LIST, 1):
        print(f"   {i}. {model}")
    
    # 训练结果记录
    results = []

    for idx, model_name in enumerate(MODEL_LIST, 1):
        print(f"\n{'='*60}")
        print(f"📊 Progress: {idx}/{len(MODEL_LIST)}")
        print(f"{'='*60}")
        
        try:
            loss, duration = train_one_model(model_name)
            
            results.append({
                "Model": model_name,
                "Final_Loss": f"{loss:.4f}",
                "Duration_Minutes": f"{duration/60:.2f}",
                "Status": "✅ Success",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"\n✅ {model_name} completed successfully!")
            print(f"   Final Loss: {loss:.4f}")
            print(f"   Duration: {duration/60:.2f} minutes")

        except Exception as e:
            print(f"\n❌ {model_name} FAILED!")
            print(f"   Error: {str(e)}")
            
            import traceback
            print("\n📋 Full traceback:")
            traceback.print_exc()
            
            results.append({
                "Model": model_name,
                "Final_Loss": "N/A",
                "Duration_Minutes": "N/A",
                "Status": f"❌ Failed: {str(e)[:100]}",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 失败后也要清理显存
            clear_gpu_memory()
            
            # 可选：是否继续下一个模型
            print("\n⚠️ Continue to next model? (y/n)")
            # 自动继续
            print("   Auto-continuing...")

    # === 保存结果 ===
    print(f"\n{'='*60}")
    print("📊 Training Summary")
    print(f"{'='*60}")
    
    df = pd.DataFrame(results)
    df.to_csv(LOG_FILE, index=False, encoding='utf-8')
    
    print(f"\n✅ Log saved to: {LOG_FILE}")
    print("\n" + "="*30)
    print(df.to_string(index=False))
    print("="*60)
    
    # 统计
    success_count = sum(1 for r in results if "Success" in r["Status"])
    print(f"\n🎉 Pipeline Complete!")
    print(f"   Successful: {success_count}/{len(MODEL_LIST)}")
    print(f"   Failed: {len(MODEL_LIST) - success_count}/{len(MODEL_LIST)}")

if __name__ == "__main__":
    main()