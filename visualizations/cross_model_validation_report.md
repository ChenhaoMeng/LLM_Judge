# Cross-Model Validation Report

## 1. Overview

This report analyzes the consistency of linguistic phenomena across different model families.
The goal is to determine whether observed patterns are universal language model limitations
or specific to particular architectures/training data.

## 2. Qwen Model Family Analysis

### 2.1 Overall Performance

| Model | Average Accuracy |
|-------|------------------|
| Qwen2.5-0.5B | 0.551 |
| Qwen2.5-1.5B | 0.592 |
| Qwen2.5-7B | 0.595 |

### 2.2 Performance by Key Phenomena

#### Ellipsis Phenomena:

| Phenomenon | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|------------|--------------|--------------|-------------|
| ellipsis | 0.506 | 0.507 | 0.512 | 

#### Anaphor Phenomena:

| Phenomenon | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|------------|--------------|--------------|-------------|
| anaphor | 0.752 | 0.742 | 0.742 | 

## 3. Scaling Analysis

### 3.1 Model Scaling Effects

As model size increases from 0.5B to 7B parameters:
- Overall accuracy improves from 55.1% to 59.5%
- Ellipsis phenomena show average improvement of 0.67%
- Anaphor phenomena show average improvement of -1.00%

## 4. Cross-Model Validation Requirements

To validate that these patterns are not specific to Qwen models:

1. **LLaMA Family**: Test with LLaMA-3-8B to see if similar patterns emerge
2. **GPT Family**: Test with GPT-2 (small) for comparison with different architecture
3. **Other Architectures**: Include models like Mistral, Mixtral, etc.

## 5. Key Findings

### 5.1 Ellipsis Analysis

- In Qwen models, ellipsis phenomena show modest improvement with scale
- This suggests that larger models may better handle ellipsis constructions
- Cross-validation with other families needed to confirm universality

### 5.2 Anaphor Analysis

- Anaphor phenomena show more complex scaling behavior
- Some anaphor types may follow U-shaped curve (improvement) while others show inverse scaling
- This requires further investigation with more model families

### 5.3 Human-Level Comparison

- Human benchmarks for BLIMP dataset are typically around 88%-90%
- Current Qwen models reach ~58% accuracy, indicating significant room for improvement
- Cross-model validation will help identify universal vs. architecture-specific limitations

## 6. Recommendations

1. Generate results for LLaMA-3-8B and GPT-2 models on the same test set
2. Compare the patterns observed in Qwen with other architectures
3. Focus on phenomena that show consistent behavior across all model families
4. Investigate phenomena that show family-specific patterns for training insights
