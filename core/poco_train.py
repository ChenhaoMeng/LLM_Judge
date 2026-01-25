"""
PoCO (Post-Correction via Overcorrection) 两阶段 SFT 训练策略

问题背景：
- 选择题（40%）表现极佳（接近 100%）
- 改错题（30%）表现较弱（0-60%）
- 这是因为大模型容易出现"过度纠错（Over-correction）"

解决方案：
- Stage 1: 使用大模型的高召回率发现错误（高敏感度）
- Stage 2: 使用微调小模型的高精确率精确筛选（高精确度）
- 通过两阶段训练平衡性能
"""

import os
import sys

# ================= 🔧 1. 环境与缓存配置 =================
os.environ['HF_HOME'] = '/mnt/drive1/chenhao/cache/huggingface'
os.environ['DISABLE_FLASH_ATTN'] = '1'

import torch
import gc
import time
import pandas as pd
import json
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# ================= ⚙️ 2. 配置 =================

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

DATA_FILES = {
    "train": os.path.join(PROJECT_ROOT, "finetune_train.jsonl"),
    "test": os.path.join(PROJECT_ROOT, "finetune_test.jsonl")
}

MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "poco_results")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "poco_training_log.csv")

# PoCO 策略配置
POCO_CONFIG = {
    "stage1_weight_mc": 0.2,      # Stage 1: 选择题权重（降低）
    "stage1_weight_judge": 0.3,   # Stage 1: 判断题权重
    "stage1_weight_correction": 0.5,  # Stage 1: 改错题权重（提升）
    
    "stage2_weight_mc": 0.4,      # Stage 2: 选择题权重（恢复）
    "stage2_weight_judge": 0.3,   # Stage 2: 判断题权重
    "stage2_weight_correction": 0.3,  # Stage 2: 改错题权重（恢复）
    
    "stage1_epochs": 2,  # Stage 1: 重点训练改错能力
    "stage2_epochs": 1,  # Stage 2: 平衡所有任务
}

# ================= 🧹 3. 显存清理工具 =================
def clear_gpu_memory():
    """强制清理显存"""
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

# ================= 📊 4. 数据集重采样与权重平衡 =================
def resample_dataset_with_weights(dataset, task_weights, task_key='instruction'):
    """
    根据任务类型对数据集进行重采样，以平衡不同任务的权重
    
    Args:
        dataset: HuggingFace Dataset
        task_weights: dict, {"mc": 0.4, "judge": 0.3, "correction": 0.3}
        task_key: str, 用于识别任务类型的字段名
    
    Returns:
        resampled_dataset: 重采样后的数据集
    """
    print(f"📊 数据集重采样 (权重: {task_weights})...")
    
    # 识别任务类型
    def identify_task_type(example):
        instruction = example.get(task_key, '').lower()
        if '选出' in instruction or '选项' in instruction:
            return 'mc'
        elif '判断' in instruction:
            return 'judge'
        elif '修改' in instruction or '修正' in instruction or '改错' in instruction:
            return 'correction'
        else:
            return 'other'
    
    # 添加任务类型标签
    dataset = dataset.map(lambda x: {"task_type": identify_task_type(x)})
    
    # 统计各类任务数量
    task_counts = {}
    for task_type in ['mc', 'judge', 'correction', 'other']:
        task_counts[task_type] = len([x for x in dataset if x['task_type'] == task_type])
    
    print(f"   原始分布: {task_counts}")
    
    # 计算目标样本数（以最少的任务为准，按权重比例）
    total_samples = sum(task_counts.values())
    min_task_samples = min([task_counts[k] for k in task_weights.keys() if k in task_counts])
    
    # 计算重采样后的目标数量
    target_counts = {}
    for task_type, weight in task_weights.items():
        if task_type in task_counts:
            target_counts[task_type] = int(total_samples * weight)
    
    print(f"   目标分布: {target_counts}")
    
    # 重采样
    resampled_examples = []
    for task_type, target_count in target_counts.items():
        task_examples = [x for x in dataset if x['task_type'] == task_type]
        
        if len(task_examples) >= target_count:
            # 如果样本数足够，随机采样
            import random
            resampled_examples.extend(random.sample(task_examples, target_count))
        else:
            # 如果样本数不足，重复采样
            import random
            resampled_examples.extend(task_examples)
            remaining = target_count - len(task_examples)
            resampled_examples.extend(random.choices(task_examples, k=remaining))
    
    # 打乱顺序
    import random
    random.shuffle(resampled_examples)
    
    # 转换为 Dataset
    resampled_dataset = Dataset.from_list(resampled_examples)
    
    print(f"   重采样后: {len(resampled_dataset)} 条样本")
    return resampled_dataset

# ================= 🚀 5. PoCO 两阶段训练 =================
def train_poco_model(model_name):
    print(f"\n{'='*60}")
    print(f"🚀 Starting PoCO training for: {model_name}")
    print(f"{'='*60}")

    short_name = model_name.split("/")[-1]
    output_dir = os.path.join(OUTPUT_ROOT, short_name)
    
    clear_gpu_memory()

    # --- A. 加载模型 ---
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

    print(f"📥 Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=model_dtype, 
        device_map="auto" if is_large_model else None, 
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    if not is_large_model:
        model = model.to("cuda:0")
    
    if is_large_model:
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False

    # --- B. 加载 Tokenizer ---
    print(f"📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # --- C. LoRA 配置 ---
    print("🔧 Applying LoRA configuration...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    if not is_large_model:
        model = model.to("cuda:0")

    # --- D. 加载和预处理数据 ---
    print("📂 Loading and processing dataset...")
    try:
        raw_datasets = load_dataset("json", data_files=DATA_FILES)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        raise

    def format_and_tokenize(example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        if input_text:
            text = f"User: {instruction}\n{input_text}\nAssistant: {output}"
        else:
            text = f"User: {instruction}\nAssistant: {output}"
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # 预处理原始数据集
    raw_datasets = raw_datasets.map(
        format_and_tokenize,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing dataset",
        batched=False,
    )

    # --- E. Stage 1: 重点训练改错能力 ---
    print(f"\n{'='*60}")
    print(f"📌 Stage 1: 重点训练改错能力（提升召回率）")
    print(f"{'='*60}")
    
    stage1_weights = {
        "mc": POCO_CONFIG["stage1_weight_mc"],
        "judge": POCO_CONFIG["stage1_weight_judge"],
        "correction": POCO_CONFIG["stage1_weight_correction"]
    }
    
    stage1_dataset = resample_dataset_with_weights(raw_datasets["train"], stage1_weights)
    
    stage1_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "stage1"),
        per_device_train_batch_size=2 if is_large_model else 4,
        per_device_eval_batch_size=2 if is_large_model else 4,
        gradient_accumulation_steps=8 if is_large_model else 4,
        learning_rate=2e-4,
        num_train_epochs=POCO_CONFIG["stage1_epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_32bit",
        gradient_checkpointing=False,
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )
    
    stage1_trainer = SFTTrainer(
        model=model,
        args=stage1_args,
        train_dataset=stage1_dataset,
        eval_dataset=raw_datasets["test"],
    )
    
    stage1_start = time.time()
    stage1_result = stage1_trainer.train()
    stage1_duration = time.time() - stage1_start
    stage1_loss = stage1_result.training_loss
    
    print(f"✅ Stage 1 completed! Final loss: {stage1_loss:.4f}, Duration: {stage1_duration/60:.2f} min")
    
    # --- F. Stage 2: 平衡所有任务 ---
    print(f"\n{'='*60}")
    print(f"📌 Stage 2: 平衡所有任务（提升精确率）")
    print(f"{'='*60}")
    
    stage2_weights = {
        "mc": POCO_CONFIG["stage2_weight_mc"],
        "judge": POCO_CONFIG["stage2_weight_judge"],
        "correction": POCO_CONFIG["stage2_weight_correction"]
    }
    
    stage2_dataset = resample_dataset_with_weights(raw_datasets["train"], stage2_weights)
    
    stage2_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "stage2"),
        per_device_train_batch_size=2 if is_large_model else 4,
        per_device_eval_batch_size=2 if is_large_model else 4,
        gradient_accumulation_steps=8 if is_large_model else 4,
        learning_rate=1e-4,  # 稍微降低学习率
        num_train_epochs=POCO_CONFIG["stage2_epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_32bit",
        gradient_checkpointing=False,
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )
    
    stage2_trainer = SFTTrainer(
        model=model,
        args=stage2_args,
        train_dataset=stage2_dataset,
        eval_dataset=raw_datasets["test"],
    )
    
    stage2_start = time.time()
    stage2_result = stage2_trainer.train()
    stage2_duration = time.time() - stage2_start
    stage2_loss = stage2_result.training_loss
    
    print(f"✅ Stage 2 completed! Final loss: {stage2_loss:.4f}, Duration: {stage2_duration/60:.2f} min")
    
    # --- G. 保存最终模型 ---
    print(f"💾 Saving final model to {output_dir}")
    final_output_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"✅ Model saved successfully to {final_output_dir}")
    
    # --- H. 清理 ---
    clear_gpu_memory()
    del model, tokenizer, stage1_trainer, stage2_trainer
    clear_gpu_memory()
    
    total_duration = stage1_duration + stage2_duration
    return stage1_loss, stage2_loss, total_duration

# ================= 🔄 6. 主循环 =================
def main():
    print("="*60)
    print("🎯 PoCO (Post-Correction via Overcorrection) Training Pipeline")
    print("="*60)
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    for split, filepath in DATA_FILES.items():
        if not os.path.exists(filepath):
            print(f"❌ Error: {split} file not found: {filepath}")
            return
        print(f"✅ Found {split} file: {filepath}")
    
    if not torch.cuda.is_available():
        print("❌ Error: No GPU available!")
        return
    
    print(f"\n🎮 GPU Info:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"\n📋 Training Queue: {len(MODEL_LIST)} models")
    for i, model in enumerate(MODEL_LIST, 1):
        print(f"   {i}. {model}")
    
    results = []
    
    for idx, model_name in enumerate(MODEL_LIST, 1):
        print(f"\n{'='*60}")
        print(f"📊 Progress: {idx}/{len(MODEL_LIST)}")
        print(f"{'='*60}")
        
        try:
            stage1_loss, stage2_loss, duration = train_poco_model(model_name)
            
            results.append({
                "Model": model_name,
                "Stage1_Loss": f"{stage1_loss:.4f}",
                "Stage2_Loss": f"{stage2_loss:.4f}",
                "Duration_Minutes": f"{duration/60:.2f}",
                "Status": "✅ Success",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"\n✅ {model_name} completed successfully!")
            print(f"   Stage 1 Loss: {stage1_loss:.4f}")
            print(f"   Stage 2 Loss: {stage2_loss:.4f}")
            print(f"   Total Duration: {duration/60:.2f} minutes")

        except Exception as e:
            print(f"\n❌ {model_name} FAILED!")
            print(f"   Error: {str(e)}")
            
            import traceback
            traceback.print_exc()
            
            results.append({
                "Model": model_name,
                "Stage1_Loss": "N/A",
                "Stage2_Loss": "N/A",
                "Duration_Minutes": "N/A",
                "Status": f"❌ Failed: {str(e)[:100]}",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            clear_gpu_memory()
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(LOG_FILE, index=False, encoding='utf-8')
    
    print(f"\n✅ Log saved to: {LOG_FILE}")
    print("\n" + "="*60)
    print(df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
