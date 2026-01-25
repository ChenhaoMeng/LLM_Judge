import json
import os
import glob
import random

# ================= ⚙️ 配置区域 =================
# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()
INPUT_DIR = os.path.join(PROJECT_ROOT, "data") 
OUTPUT_DIR = PROJECT_ROOT

# 🔥 修复数据泄露：使用按句子对分割的策略
# 默认生成新文件，避免覆盖旧数据
USE_FIXED_NAMES = True  # 设为False可覆盖旧文件

if USE_FIXED_NAMES:
    TRAIN_FILE = os.path.join(OUTPUT_DIR, "finetune_multitask_train_fixed.jsonl")
    TEST_FILE = os.path.join(OUTPUT_DIR, "finetune_multitask_test_fixed.jsonl")
else:
    TRAIN_FILE = os.path.join(OUTPUT_DIR, "finetune_multitask_train.jsonl")
    TEST_FILE = os.path.join(OUTPUT_DIR, "finetune_multitask_test.jsonl")

TEST_RATIO = 0.3
RANDOM_SEED = 42

# 任务比例权重 (选择题 : 判断题 : 改错题)
TASK_WEIGHTS = [0.4, 0.3, 0.3]

# 🔥 新增：Explain-then-Process (EtP) 开关
# EtP 要求模型在改错题中先解释错误原因，再输出修正结果
# 这有助于模型真正理解语法规则，而不仅仅是记住答案模式
USE_ETP = True  # 设为 False 可禁用 EtP，使用原始格式 

# 针对不同现象的 Prompt 模板池
PROMPT_TEMPLATES = {
    "anaphor": {
        "mc": "请分析代词指代关系，选出正确的句子。",
        "judge": "请判断以下句子中代词与先行词的指代关系是否正确。",
        "correct": "以下句子中的代词指代有误，请进行修改，使其指代明确且性数一致。",
        "correct_etp": "以下句子中的代词指代有误。请先分析错误原因，然后进行修改，使其指代明确且性数一致。"
    },
    "ellipsis": {
        "mc": "请选出省略用法自然、符合语法的句子。",
        "judge": "请判断以下句子的省略结构是否符合中文习惯。",
        "correct": "以下句子的省略造成了歧义或不通顺，请将其修改为自然的句子。",
        "correct_etp": "以下句子的省略造成了歧义或不通顺。请先分析省略不当的原因，然后将其修改为自然的句子。"
    },
    "quantifiers": {
        "mc": "请选出量词逻辑语义通顺的句子。",
        "judge": "请判断以下句子中量词的使用是否符合逻辑。",
        "correct": "以下句子的量词逻辑不通顺，请进行修改。",
        "correct_etp": "以下句子的量词逻辑不通顺。请先说明量词使用不当的原因，然后进行修改。"
    },
    "default": {
        "mc": "请选出语法正确的句子。",
        "judge": "请判断以下句子是否符合语法规范。",
        "correct": "以下句子存在语法错误，请予以修正。",
        "correct_etp": "以下句子存在语法错误。请先分析错误原因，然后予以修正。"
    }
}

# ================= 🚀 核心逻辑 =================

def get_task_type(filename):
    if "anaphor" in filename: return "anaphor"
    if "ellipsis" in filename: return "ellipsis"
    if "quantifiers" in filename: return "quantifiers"
    return "default"

def create_multitask_entries(item, task_type, generate_all_formats=False):
    """
    输入一行原始数据，生成所有任务格式的数据（用于修复数据泄露）
    
    Args:
        item: 原始数据项，包含 sentence_good 和 sentence_bad
        task_type: 任务类型（anaphor/ellipsis/quantifiers）
        generate_all_formats: 如果为True，生成所有三种格式；如果为False，随机生成一种（旧逻辑）
    """
    good = item.get('sentence_good')
    bad = item.get('sentence_bad')
    templates = PROMPT_TEMPLATES[task_type]
    
    entries = []
    
    if generate_all_formats:
        # 🔥 修复数据泄露：生成所有三种格式，确保同一句子对的所有格式都在同一集合中
        
        # --- 模式 0: 选择题 (Multiple Choice) ---
        options = [good, bad]
        random.shuffle(options)  # 随机打乱选项顺序
        correct = "选项 A" if options[0] == good else "选项 B"
        entries.append({
            "instruction": templates["mc"],
            "input": f"选项 A: {options[0]}\n选项 B: {options[1]}",
            "output": f"{correct} 是正确的。",
            "_sentence_pair": tuple(sorted([good, bad]))  # 用于分组
        })

        # --- 模式 1: 判断题 (Judgment) - 生成两个（好句子和坏句子各一个）---
        # 判断好句子
        entries.append({
            "instruction": templates["judge"],
            "input": f"句子：{good}",
            "output": "是，该句子语法正确且逻辑通顺。",
            "_sentence_pair": tuple(sorted([good, bad]))
        })
        # 判断坏句子
        entries.append({
            "instruction": templates["judge"],
            "input": f"句子：{bad}",
            "output": "否，该句子存在语法或逻辑问题。",
            "_sentence_pair": tuple(sorted([good, bad]))
        })

        # --- 模式 2: 改错题 (Correction) ---
        if USE_ETP:
            error_explanation = generate_error_explanation(bad, good, task_type)
            entries.append({
                "instruction": templates.get("correct_etp", templates["correct"]),
                "input": f"错误句子：{bad}",
                "output": f"错误原因：{error_explanation}\n修正后的句子：{good}",
                "_sentence_pair": tuple(sorted([good, bad]))
            })
        else:
            entries.append({
                "instruction": templates["correct"],
                "input": f"错误句子：{bad}",
                "output": f"修正后的句子：{good}",
                "_sentence_pair": tuple(sorted([good, bad]))
            })
    else:
        # 旧逻辑：随机生成一种格式（保持向后兼容）
        mode = random.choices([0, 1, 2], weights=TASK_WEIGHTS, k=1)[0]
        
        if mode == 0:
            options = [good, bad]
            random.shuffle(options)
            correct = "选项 A" if options[0] == good else "选项 B"
            entries.append({
                "instruction": templates["mc"],
                "input": f"选项 A: {options[0]}\n选项 B: {options[1]}",
                "output": f"{correct} 是正确的。"
            })
        elif mode == 1:
            is_testing_good = random.choice([True, False])
            target_sent = good if is_testing_good else bad
            answer = "是，该句子语法正确且逻辑通顺。" if is_testing_good else "否，该句子存在语法或逻辑问题。"
            entries.append({
                "instruction": templates["judge"],
                "input": f"句子：{target_sent}",
                "output": answer
            })
        elif mode == 2:
            if USE_ETP:
                error_explanation = generate_error_explanation(bad, good, task_type)
                entries.append({
                    "instruction": templates.get("correct_etp", templates["correct"]),
                    "input": f"错误句子：{bad}",
                    "output": f"错误原因：{error_explanation}\n修正后的句子：{good}"
                })
            else:
                entries.append({
                    "instruction": templates["correct"],
                    "input": f"错误句子：{bad}",
                    "output": f"修正后的句子：{good}"
                })
        
    return entries

def generate_error_explanation(bad_sentence, good_sentence, task_type):
    """
    根据语法现象类型生成错误原因说明
    
    这是一个辅助函数，用于生成错误解释。在实际应用中，
    可以通过规则或模板匹配生成更精确的解释。
    """
    explanations = {
        "anaphor": f"代词与先行词在性别或数量上不一致。",
        "ellipsis": f"省略结构不当，导致语义不清或不符合中文表达习惯。",
        "quantifiers": f"量词与名词的搭配不符合逻辑或语义规范。",
        "default": f"存在语法错误或逻辑问题。"
    }
    
    # 可以进一步细化，例如：
    # - 对于 anaphor：分析具体的性别/数量不一致
    # - 对于 ellipsis：说明省略了什么导致问题
    # - 对于 quantifiers：说明量词与名词不匹配的具体原因
    
    base_explanation = explanations.get(task_type, explanations["default"])
    
    # 可以添加更详细的解释（可选）
    # 这里提供一个基础版本，实际使用时可以根据需要扩展
    return base_explanation

def process_files():
    random.seed(RANDOM_SEED)
    
    print(f"🔍 正在搜索数据文件: {INPUT_DIR}/*.jsonl")
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 数据目录不存在 -> {INPUT_DIR}")
        return

    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    # 排除输出文件
    files = [f for f in files if "finetune_" not in f]
    
    if not files:
        print(f"❌ 错误: 在 {INPUT_DIR} 下没有找到 .jsonl 文件！")
        return

    # 🔥 修复数据泄露：按句子对分组
    print(f"📂 发现 {len(files)} 个文件，开始生成多任务混合数据集...")
    print("🔧 使用修复后的数据分割策略：按句子对分割，避免跨格式泄露")
    
    # 第一步：收集所有原始句子对
    sentence_pairs = {}  # {(good, bad): [entries]}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        task_type = get_task_type(filename)
        count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if not item.get('sentence_good') or not item.get('sentence_bad'):
                        continue
                    
                    good = item.get('sentence_good')
                    bad = item.get('sentence_bad')
                    # 使用排序后的元组作为key，确保同一句子对只出现一次
                    pair_key = tuple(sorted([good, bad]))
                    
                    # 为每个句子对生成所有格式的任务
                    new_entries = create_multitask_entries(item, task_type, generate_all_formats=True)
                    
                    if pair_key not in sentence_pairs:
                        sentence_pairs[pair_key] = []
                    sentence_pairs[pair_key].extend(new_entries)
                    count += 1
                    
                except Exception as e:
                    print(f"  ⚠️ 处理 {filename} 时出错: {e}")
                    continue
        print(f"  - 处理 {filename}: {count} 个句子对")

    if not sentence_pairs:
        print("❌ 错误: 处理后数据为空！请检查源文件内容格式。")
        return

    print(f"\n📊 统计信息:")
    print(f"   - 唯一句子对数: {len(sentence_pairs)}")
    total_entries = sum(len(entries) for entries in sentence_pairs.values())
    print(f"   - 总样本数: {total_entries}")
    print(f"   - 平均每个句子对生成 {total_entries/len(sentence_pairs):.1f} 个样本")

    # 第二步：按句子对分割训练集和测试集
    pairs_list = list(sentence_pairs.keys())
    random.shuffle(pairs_list)
    
    split_idx = int(len(pairs_list) * (1 - TEST_RATIO))
    train_pairs = set(pairs_list[:split_idx])
    test_pairs = set(pairs_list[split_idx:])
    
    print(f"\n📊 数据分割:")
    print(f"   - 训练集句子对数: {len(train_pairs)} ({len(train_pairs)/len(pairs_list)*100:.1f}%)")
    print(f"   - 测试集句子对数: {len(test_pairs)} ({len(test_pairs)/len(pairs_list)*100:.1f}%)")
    
    # 第三步：重新组织数据
    train_data = []
    test_data = []
    
    for pair_key, entries in sentence_pairs.items():
        # 移除内部标记字段
        clean_entries = []
        for entry in entries:
            clean_entry = {k: v for k, v in entry.items() if not k.startswith('_')}
            clean_entries.append(clean_entry)
        
        if pair_key in train_pairs:
            train_data.extend(clean_entries)
        else:
            test_data.extend(clean_entries)
    
    # 打乱每个集合内的顺序（但保持句子对级别的分离）
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f"\n📊 最终数据统计:")
    print(f"   - 训练集样本数: {len(train_data)}")
    print(f"   - 测试集样本数: {len(test_data)}")
    
    # 验证：检查是否有泄露
    train_sentences = set()
    test_sentences = set()
    
    for entry in train_data:
        input_text = entry.get('input', '')
        # 提取所有句子
        if '选项 A:' in input_text:
            for line in input_text.split('\n'):
                if '选项' in line and ':' in line:
                    sent = line.split(':', 1)[1].strip()
                    if sent:
                        train_sentences.add(sent)
        elif '句子：' in input_text:
            sent = input_text.split('句子：', 1)[1].strip()
            if sent:
                train_sentences.add(sent)
        elif '错误句子：' in input_text:
            sent = input_text.split('错误句子：', 1)[1].strip()
            if sent:
                train_sentences.add(sent)
    
    for entry in test_data:
        input_text = entry.get('input', '')
        if '选项 A:' in input_text:
            for line in input_text.split('\n'):
                if '选项' in line and ':' in line:
                    sent = line.split(':', 1)[1].strip()
                    if sent:
                        test_sentences.add(sent)
        elif '句子：' in input_text:
            sent = input_text.split('句子：', 1)[1].strip()
            if sent:
                test_sentences.add(sent)
        elif '错误句子：' in input_text:
            sent = input_text.split('错误句子：', 1)[1].strip()
            if sent:
                test_sentences.add(sent)
    
    overlap = train_sentences & test_sentences
    print(f"\n✅ 数据泄露检查:")
    print(f"   - 训练集唯一句子数: {len(train_sentences)}")
    print(f"   - 测试集唯一句子数: {len(test_sentences)}")
    print(f"   - 重叠句子数: {len(overlap)}")
    if len(test_sentences) > 0:
        leak_ratio = len(overlap) / len(test_sentences) * 100
        print(f"   - 泄露比例: {leak_ratio:.2f}%")
        if leak_ratio > 0:
            print(f"   ⚠️ 警告：仍有 {len(overlap)} 个句子重叠！")
        else:
            print(f"   ✅ 未发现句子级别的泄露")
    
    # 保存
    def save(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    
    save(train_data, TRAIN_FILE)
    save(test_data, TEST_FILE)
    
    print("-" * 60)
    print(f"✅ 完成！数据已保存")
    print(f"   - 训练集: {len(train_data)} 条 -> {TRAIN_FILE}")
    print(f"   - 测试集: {len(test_data)} 条 -> {TEST_FILE}")
    
    # 安全打印样例
    print("\n🔍 训练集样例展示:")
    preview_count = min(3, len(train_data))
    for i in range(preview_count):
        sample = train_data[i].copy()
        print(f"Sample {i+1}: {json.dumps(sample, ensure_ascii=False)}")

if __name__ == "__main__":
    process_files()