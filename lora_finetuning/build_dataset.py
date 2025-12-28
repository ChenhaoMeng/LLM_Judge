import json
import pandas as pd
from datasets import Dataset
import os

def build_finetuning_dataset():
    """
    构建针对 Ellipsis 和 Anaphor 的微调数据集
    """
    # 创建针对 Ellipsis 和 Anaphor 的数据
    # 基于 BLIMP 数据集的模式生成示例数据
    ellipsis_data = [
        {
            "text": "John said Mary would come, and she did come.",
            "label": "ellipsis",
            "target": "correct"
        },
        {
            "text": "John said Mary would come, and she didn't.",
            "label": "ellipsis", 
            "target": "incorrect"
        },
        {
            "text": "Bill claimed Sarah would run, and she did run.",
            "label": "ellipsis",
            "target": "correct"
        },
        {
            "text": "Bill claimed Sarah would run, and she didn't run.",
            "label": "ellipsis",
            "target": "incorrect"
        },
        {
            "text": "Tom believed Jane would dance, and she did dance.",
            "label": "ellipsis",
            "target": "correct"
        },
        {
            "text": "Tom believed Jane would dance, and she didn't dance.",
            "label": "ellipsis",
            "target": "incorrect"
        },
        {
            "text": "The men said the women would sing, and they did sing.",
            "label": "ellipsis",
            "target": "correct"
        },
        {
            "text": "The men said the women would sing, and they didn't sing.",
            "label": "ellipsis",
            "target": "incorrect"
        },
        {
            "text": "Alice thought Bob would study, and he did study.",
            "label": "ellipsis",
            "target": "correct"
        },
        {
            "text": "Alice thought Bob would study, and he didn't study.",
            "label": "ellipsis",
            "target": "incorrect"
        }
    ]
    
    anaphor_data = [
        {
            "text": "The cat saw itself in the mirror.",
            "label": "anaphor",
            "target": "correct"
        },
        {
            "text": "The cat saw itself in the mirrors.",  # 错误：数的一致性
            "label": "anaphor",
            "target": "incorrect"
        },
        {
            "text": "The dog chased itself around the yard.",
            "label": "anaphor",
            "target": "correct"
        },
        {
            "text": "The dog chased itself around the yards.",  # 错误：数的一致性
            "label": "anaphor",
            "target": "incorrect"
        },
        {
            "text": "The children hurt themselves while playing.",
            "label": "anaphor",
            "target": "correct"
        },
        {
            "text": "The children hurt themselves while playing games.",  # 正确但更复杂
            "label": "anaphor",
            "target": "correct"
        },
        {
            "text": "The boy saw himself in the reflection.",
            "label": "anaphor",
            "target": "correct"
        },
        {
            "text": "The boy saw himself in the reflections.",  # 错误：数的一致性
            "label": "anaphor",
            "target": "incorrect"
        },
        {
            "text": "The students taught themselves mathematics.",
            "label": "anaphor",
            "target": "correct"
        },
        {
            "text": "The student taught themselves mathematics.",  # 错误：数的一致性
            "label": "anaphor",
            "target": "incorrect"
        }
    ]
    
    # 合并数据
    all_data = ellipsis_data + anaphor_data
    
    # 创建 HuggingFace Dataset
    dataset = Dataset.from_list(all_data)
    
    # 保存数据集
    dataset.save_to_disk("/workspace/lora_finetuning/data/syntactic_dataset")
    
    # 也保存为 JSON 格式用于其他用途
    df = pd.DataFrame(all_data)
    df.to_json("/workspace/lora_finetuning/data/syntactic_dataset.json", orient='records', indent=2)
    
    print(f"Dataset created with {len(all_data)} samples")
    print(f"  - Ellipsis samples: {len(ellipsis_data)}")
    print(f"  - Anaphor samples: {len(anaphor_data)}")
    print("Dataset saved to /workspace/lora_finetuning/data/")
    
    return dataset

if __name__ == "__main__":
    build_finetuning_dataset()