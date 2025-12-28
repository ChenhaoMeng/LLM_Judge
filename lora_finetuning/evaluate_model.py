import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

def load_base_model(model_name="Qwen/Qwen2.5-0.5B"):
    """
    加载基础模型
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    return model, tokenizer

def load_finetuned_model(base_model_name, finetuned_path):
    """
    加载微调后的模型
    """
    base_model, tokenizer = load_base_model(base_model_name)
    
    # 加载 LoRA 微调权重
    model = PeftModel.from_pretrained(base_model, finetuned_path)
    
    return model, tokenizer

def evaluate_sentence_correctness(model, tokenizer, sentences, targets):
    """
    评估句子正确性
    """
    model.eval()
    correct_predictions = 0
    total = len(sentences)
    
    results = []
    
    for i, (sentence, target) in enumerate(zip(sentences, targets)):
        # 构建提示
        prompt = f"Analyze the following sentence for syntactic correctness: {sentence}\nIs it grammatically correct? Answer:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip().lower()
        
        # 判断预测结果
        predicted_correct = "correct" in response or "yes" in response
        actual_correct = target == "correct"
        
        is_correct = predicted_correct == actual_correct
        if is_correct:
            correct_predictions += 1
        
        results.append({
            'sentence': sentence,
            'target': target,
            'prediction': 'correct' if predicted_correct else 'incorrect',
            'response': response,
            'is_correct': is_correct
        })
        
        print(f"Example {i+1}:")
        print(f"  Sentence: {sentence}")
        print(f"  Target: {target}")
        print(f"  Prediction: {'correct' if predicted_correct else 'incorrect'}")
        print(f"  Model response: {response}")
        print(f"  Correct: {is_correct}")
        print()
    
    accuracy = correct_predictions / total
    print(f"Overall accuracy: {accuracy:.2%} ({correct_predictions}/{total})")
    
    return results, accuracy

def main():
    # 测试句子（与训练数据类似但不完全相同）
    test_sentences = [
        ("John said Mary would come, and she did come.", "correct"),
        ("John said Mary would come, and she didn't.", "incorrect"),
        ("The cat saw itself in the mirror.", "correct"),
        ("The cat saw itself in the mirrors.", "incorrect"),
        ("Bill claimed Sarah would run, and she did run.", "correct"),
        ("The dog chased itself around the yards.", "incorrect"),
        ("Tom believed Jane would dance, and she didn't dance.", "incorrect"),
        ("The children hurt themselves while playing.", "correct"),
    ]
    
    sentences, targets = zip(*test_sentences)
    
    print("Evaluating base model...")
    base_model, base_tokenizer = load_base_model("Qwen/Qwen2.5-0.5B")
    base_results, base_accuracy = evaluate_sentence_correctness(
        base_model, base_tokenizer, sentences, targets
    )
    
    print("\n" + "="*50 + "\n")
    
    # 尝试加载微调后的模型（如果存在）
    try:
        print("Evaluating fine-tuned model...")
        ft_model, ft_tokenizer = load_finetuned_model(
            "Qwen/Qwen2.5-0.5B", 
            "/workspace/lora_finetuning/fine_tuned_model"
        )
        ft_results, ft_accuracy = evaluate_sentence_correctness(
            ft_model, ft_tokenizer, sentences, targets
        )
        
        print(f"\nImprovement: {ft_accuracy - base_accuracy:.2%}")
        
    except Exception as e:
        print(f"Could not load fine-tuned model (may not exist yet): {e}")
        print("Run the lora_finetuning.py script first to create the fine-tuned model.")

if __name__ == "__main__":
    main()