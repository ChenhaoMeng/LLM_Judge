import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def prepare_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B"):
    """
    加载模型和tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    return model, tokenizer

def prepare_dataset_for_training(dataset_path, tokenizer):
    """
    准备训练数据集
    """
    dataset = load_from_disk(dataset_path)
    
    def tokenize_function(examples):
        # 创建输入文本
        texts = []
        labels = []
        
        for i in range(len(examples['text'])):
            text = examples['text'][i]
            target = examples['target'][i]
            
            # 构建提示模板
            prompt = f"Analyze the following sentence for syntactic correctness: {text}\nIs it grammatically correct? Answer: {target}"
            texts.append(prompt)
            labels.append(1 if target == "correct" else 0)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 对于 causal LM，labels 与 input_ids 相同（除了我们不想计算损失的部分）
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 应用 tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def setup_lora_config():
    """
    设置 LoRA 配置
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    return config

def compute_metrics(eval_pred):
    """
    计算评估指标
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 设置模型和tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"  # 可以改为 "Qwen/Qwen2.5-1.5B" 如果需要更大的模型
    print(f"Loading model: {model_name}")
    
    model, tokenizer = prepare_model_and_tokenizer(model_name)
    
    # 准备数据集
    print("Preparing dataset...")
    dataset = prepare_dataset_for_training("/workspace/lora_finetuning/data/syntactic_dataset", tokenizer)
    
    # 设置 LoRA
    print("Setting up LoRA...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 显示可训练参数
    model.print_trainable_parameters()
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="/workspace/lora_finetuning/results",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # 根据显存调整
        per_device_eval_batch_size=2,
        warmup_steps=5,
        weight_decay=0.01,
        logging_dir="/workspace/lora_finetuning/logs",
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # 禁用wandb等报告工具
    )
    
    # 创建 trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # 使用相同数据集进行评估（实际应用中应分开）
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    model.save_pretrained("/workspace/lora_finetuning/fine_tuned_model")
    tokenizer.save_pretrained("/workspace/lora_finetuning/fine_tuned_model")
    
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main()