# Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging

source: https://arxiv.org/html/2412.19512v1
by Hua Farn

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
  - [2.1 Jailbreaking LLMs through Fine-Tuning](#21-jailbreaking-llms-through-fine-tuning)
  - [2.2 Model Merging](#22-model-merging)
- [3 Methodology](#3-methodology)
- [4 Experimental Setups](#4-experimental-setups)
  - [4.1 Downstream Tasks](#41-downstream-tasks)
  - [4.2 Safety Evaluation](#42-safety-evaluation)
  - [4.3 Large Language Models](#43-large-language-models)
  - [4.4 Merging Methods](#44-merging-methods)
- [5 Results and Discussion](#5-results-and-discussion)
- [6 Conclusion](#6-conclusion)
- [7 Limitations](#7-limitations)
- [8 Ethics Statement](#8-ethics-statement)
- [Appendix A Domain-Specific Tasks Detail](#appendix-a-domain-specific-tasks-detail)
- [Appendix B Safety Dataset and Classifier](#appendix-b-safety-dataset-and-classifier)
  - [B.1 Safety Dataset](#b1-safety-dataset)
  - [B.2 Safety Classifier](#b2-safety-classifier)
- [Appendix C Experimental Detail](#appendix-c-experimental-detail)
  - [C.1 Prompt Template](#c1-prompt-template)
  - [C.2 Fine-tuning](#c2-fine-tuning)
  - [C.3 Inference](#c3-inference)
- [Appendix D Model Merging](#appendix-d-model-merging)
  - [D.1 Merging Methods](#d1-merging-methods)
  - [D.2 Model Merging Implementation](#d2-model-merging-implementation)
- [Appendix E Impact of «math»λ«/math» for Safety](#appendix-e-impact-of-mathλmath-for-safety)

## Abstract

**Merging Weights of Pre- and Post-Fine-Tuned Safety-Aligned Models**

**Background:**
- Fine-tuning large language models (LLMs) for downstream tasks can lead to safety degradation in safety-aligned LLMs
- Incorporating additional safety data is often impractical

**Problem Statement:**
- How can we improve downstream task performance while preserving safety in LLMs without relying on additional safety data?

**Proposed Method:**
- Merge the weights of pre- and post-fine-tuned safety-aligned models
- Demonstrates effective mitigation of safety degradation while improving downstream task performance

**Approach:**
- Pre and post fine-tuning model merging
- Maintains inherent safety of LLMs
- Practical solution for adapting safety-aligned LLMs

**Experimental Results:**
- Across various downstream tasks, models, and merging methods
- Effectively mitigates safety degradation while improving downstream task performance

**Authors:**
- Hua Farn, Hsuan Su, Shachi H Kumar, Saurav Sahay, Shang-Tse Chen, Hung-yi Lee, and alhena.farn@gmail.com (National Taiwan University & Intel Lab)

## 1 Introduction

**LLM Safety Challenges:**
- Critical focus on aligning LLMs with human values and cultural norms [^18]
- Preference tuning introduced for safety [^25]
- Base models used for downstream tasks via Supervised Fine-Tuning (SFT) [^6]

**Challenges with Fine-Tuning:**
- Safety degradation [^37]
  * Base models can generate harmful content after fine-tuning on benign datasets [^27]
  * Additional safety data during fine-tuning or complex re-alignment processes required [^12]
  * Resource-intensive and constrained by scarcity of safety data

**Proposed Approach:**
1. Fine-tune base model on downstream task
2. Merge fine-tuned model with base model
3. Evaluated across various models, tasks, merging techniques, and safety benchmarks [^35]
4. Enhances downstream task performance while significantly preserving model safety [Key Contributions]
   - Reduces Attack Success Rate (ASR) up to 30%
   - Demonstrates robustness in preserving model safety through comprehensive evaluation

## 2 Related Work

### 2.1 Jailbreaking LLMs through Fine-Tuning

Despite efforts to ensure LLMs' [^25] safety, "jailbreaking" attacks and fine-tuning issues remain threats [^36][^37]. A common mitigation involves adding safety data during fine-tuning [^27]. However, this is limited by data availability and computational costs. We propose a simple method that merges the base model with the fine-tuned model to mitigate these challenges, without requiring additional resources.

### 2.2 Model Merging





Model merging combines multiple models into one unified model. Techniques like SLERP and Model Stock enhance this process. Task vectors can also be used to enable flexibility through arithmetic operations. However, our method leverages the inherent safety of the base model, reducing the need for additional safety data.

## 3 Methodology

**Approach for Maintaining Safety of Fine-Tuned LLMs:**
* Two-step process: fine-tuning + merging
* Step 1: Fine-tune base model on downstream tasks
	+ Instruction `x^t` and response `y^t` given
	+ Optimize language model (f) with following loss function:
	```
	-log fθ(yt|xt)
	```
* Step 2: Merge fine-tuned model with base model
	+ Interpolation of parameters using formula:
	```
	θ_merged=(1-λ)θ_base+λθ_t
	```
	+ Normalize ratios to ensure sum equals 1
	+ Experiment with various merging techniques, including parameter adjustment before merging and optimal λ identification strategies.

## 4 Experimental Setups

We address four research questions through this experimental setup:
Q1: Can merging fine-tuned models with their base models prevent safety degradation?
Q2: How do different merging methods perform?
Q3: What is the impact of λ on downstream task performance and safety?
Q4: Does model merging work across multiple models?

### 4.1 Downstream Tasks

**Method Evaluation**
- Four downstream tasks: reasoning, medical assistance, code generation, tool usage proficiency
- **Reasoning**:
    - Improved using Chain-of-Thought data from Flan V2 dataset
    - Evaluated on Big Bench Hard (BBH) dataset
- **Medical Assistance**:
    - Uses patient-doctor interactions from ChatDoctor dataset
    - Measured using BERTScore with embedding extraction from 40th layer of microsoft/deberta-xlarge-mnli
- **Code Generation**:
    - Uses MagiCoder dataset for assessment
    - Evaluated via HumanEval dataset
- **Tool Usage Proficiency**:
    - Uses OpenFunctions dataset for API call generation
    - Measured using BERTScore with embedding extraction from 40th layer of microsoft/deberta-xlarge-mnli

**Additional Details:**
- See Appendix A for more details on the downstream tasks.

### 4.2 Safety Evaluation

Safety is assessed using 850 harmful instructions from AdvBench (520) and HEx-PHI (330) datasets. The WildGuard model evaluates safety, similar to GPT-4, using ASR metric for both datasets.

### 4.3 Large Language Models

We evaluate two LLM families: Llama-3 and Gemma, using their instruct-tuned versions. Within Llama-3, we test 8B-Instruct and 3.1-8B-Instruct models. For Gemma, we use the 2B-it model, fine-tuning each with three seeds via LoRA for downstream tasks.

### 4.4 Merging Methods

We evaluate three merging methods: Linear Merging, Model Stock, SLERP, and DARE. For each method, models are merged with a base model using weights selected based on validation set performance. More details can be found in Appendix D.

## 5 Results and Discussion

**Table 1: Results for Q1**
- Average scores of Llama-3-8B-Instruct models fine-tuned with different seeds for FT and Linear Merging
- Performance indicates task effectiveness measured by respective metrics
- Fine-tuned model demonstrates improved performance but compromises safety
- Merging fine-tuned model with base model enhances safety across all downstream tasks

**Table 2: Results for Q2**
- Different merging methods beneficial for the safety and task performance of fine-tuned models
- SLERP, DARE, and Linear Merging reduce ASR on AdvBench and HEx-PHI
- Linear Merging demonstrates strong performance and can be a viable alternative for practical applications

**Figure 2: Results for Q3**
- Impact of λ on downstream task performance and safety across three merging methods
- As λ increases, performance improves but ASR rises, suggesting a trade-off between performance and safety
- Optimal λ around 0.5∼0.6 for evenly combining weights and maintaining safety
- Linear Merging has slower rate of ASR increase than SLERP and DARE, making it more practical

**Table 3: Results for Q4**
- Llama-3.1-8B-Instruct and Gemma-2B-it models demonstrate mild safety degradation after fine-tuning
- Model merging restores safety without significantly compromising downstream task performance, demonstrating the method's applicability across different LLMs.

## 6 Conclusion

We propose a two-step method to address safety degradation in aligned LLMs by merging pre- and post-fine-tuned weights. This approach preserves original safety features while acquiring new task capabilities without additional safety data, as shown across various tasks, models, and techniques.

## 7 Limitations

**Task and Model Selection**
- Evaluation on benign data from: reasoning, medical assistance, code generation, tool-using proficiency tasks
- Exclusion of other domains (law, finance) for investigation
- Section [5] examines methods' efficacy on these four downstream tasks
- Performance of aligned models fine-tuned on other domains, languages, or contaminated datasets is uncertain and warrants further exploration
- Testing limited to 2B and 8B models from two model families
- Efficacy on larger models and different model families is unknown

**Safety Classifier for Safety Evaluation**
- Use of WildGuard [^11] as an alternative approach for classifying model responses as safe or unsafe instead of LLM-as-Judge [^5] with GPT-4 [^24]
- Reduced costs but limitations: struggles with complex instructions, may produce false positives or negatives, provides less detailed evaluations
- Unable to analyze which types of harmful instructions models are vulnerable to or effectively defended against after applying the method
- More detailed safety analysis is left for future work.

## 8 Ethics Statement

Our method addresses safety degradation without additional data, but merging pre- and post-fine-tuned models may inherit latent biases from the base model. Further investigation is needed to assess this impact.

## Appendix A Domain-Specific Tasks Detail

**Performance Evaluation**
- **Reasoning**:
  - Select 10,000 zero-shot chain-of-thought instructions from Flan V2 dataset
  - Split into training set (90%) and validation set (10%)
  - Evaluate using BBH dataset
  - Average 3-shot accuracy across all BBH tasks
  - Use lm-evaluation-harness for code base

- **Medical Assistance**:
  - Select 10,000 real patient-doctor conversations from ChatDoctor dataset
  - Split into training set (90%) and validation set (10%)
  - Evaluate on 1,000 unseen patient queries
  - Use BERTScore to calculate similarity between reference responses and models' responses
  - Report F1 score as performance metric

- **Code Generation**:
  - Select 10,000 samples from MagiCoder dataset to improve code generation capabilities
  - Uniformly sample from each coding language
  - Report Pass@10 in experiment results

- **Tool Using Proficiency**:
  - Use smaller OpenFunctions dataset
  - Split full training set into two subsets (9:1 ratio) for training and validating
  - Evaluate on full OpenFunctions test set
  - Use BERTScore to calculate similarity between reference responses and models' responses
  - Report F1 score as performance metric

## Appendix B Safety Dataset and Classifier

### B.1 Safety Dataset

AdvBench contains 520 harmful behaviors. HEx-PHI has 330 instructions across 11 prohibited categories, like child abuse content and hate speech.

### B.2 Safety Classifier

**WildGuard Model**:
- Achieves three goals:
  1. Detection of harm in user prompts
  2. Detection of harm in LLM responses
  3. Refusal evaluation of LLM responses

**Performance**:
- Demonstrates outperformance over existing open-source baselines in F1 scores
- Matches GPT-4 across various tasks

**Evaluation Process**:
1. Apply recommended settings, such as instruction format and decoding strategy
2. Evaluate LLM responses to harmful instructions using WildGuard
3. Output response in the following format: "We then parse the result of the harmful response. If the result cannot be parsed, we count it as a miss."
4. Calculate final **ASR** (Automatic Safety Response) based on:
   - R_yes: number of harmful responses classified as "yes"
   - R_total: total number of responses
   - R_miss: number of responses that failed to be parsed

**Experiment Findings**:
- **R_miss** is usually less than 5 for all tested models across both safety datasets.

## Appendix C Experimental Detail

### C.1 Prompt Template

We apply base models' own templates during training and inference. For fine-tuned models, we use their base model's templates. 

For Llama-3 family:
- Reasoning, code generation, and tool usage: "You are a helpful assistant."
 
For medical assistance (ChatDoctor dataset):
- The system prompt is provided separately.

Gemma-2B-it for reasoning, code generation, and tool usage has the following template:

The prompt for the medical assistance task is as follows:

### C.2 Fine-tuning





"We fine-tune three models (seed: 42, 1024, 48763) using LoRA (r=8, α=16) and AdamW optimizer (lr=1e-4). We train on an RTX A6000 or RTX 6000 GPU for 3 epochs. Due to observed improved performance, we report models trained after 500 steps for reasoning and code generation, and 200 steps for tool usage proficiency."

### C.3 Inference

We use greedy decoding for consistency except in HumanEval. There, sampling-based decoding is applied with specific parameters to achieve faster inference using the VLLM engine.

## Appendix D Model Merging

### D.1 Merging Methods

**Linear Merging**
- Involves directly combining base model and fine-tuned model weights by interpolating their parameters
- Calculates merged model weights as weighted average of base and fine-tuned models' weights (Equation [3])
- Popular choice for basic integration due to being computationally efficient

**SLERP (Spherical Linear Interpolation)**
- Advanced merging technique that interpolates between model weights on a hypersphere
- Accounts for angular relationship between weight vectors
- Preserves base model's features while integrating fine-tuned model's enhancements

**DARE (Drop and Rescale)**
- Prepares models for merging techniques like Linear Merging
- Randomly drops parameters according to drop rate and rescales remaining ones
- Helps reduce redundant or interfering parameters among multiple models

**Model Stock**
- Uses geometric properties of fine-tuned model weights for optimal merging ratio determination
- Requires at least two fine-tuned models and a base model
- Merges three fine-tuned models uniformly, then merges average with base model using Model Stock's optimal ratio.

### D.2 Model Merging Implementation

We test interpolation weights λ in the set {0.2, 0.4, 0.6, 0.8} for Linear Merging, SLERP, and DARE algorithms. Model Stock does not require hyperparameter specification due to its automatic weight approximation feature. We use MergeKit as our codebase.

## Appendix E Impact of «math»λ«/math» for Safety

The impact of λ on AdvBench is shown in Figure 3. Like HEx-PHI, ASR increases as λ increases, but remains better than SFT's results.

