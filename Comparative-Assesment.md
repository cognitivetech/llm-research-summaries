# Finetuning LLMs for Comparative Assessment Tasks

https://arxiv.org/pdf/2409.15979

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related work](#2-related-work)
- [3 LLM comparative assessment](#3-llm-comparative-assessment)
- [4 Experiments](#4-experiments)
- [5 Results](#5-results)
- [6 Conclusions](#6-conclusions)
- [7 Limitations](#7-limitations)

## Abstract

**Comparison of Large Language Models (LLMs) for Natural Language Generation Assessment**

**Challenges of Automated Assessment:**
- Difficult task in natural language generation
- Instruction-tuned LLMs show promise through comparative assessment

**Limitations of Current Approaches:**
- Quadratic computational complexity in pairwise comparisons limits scalability
- Efficient comparative assessment explored to address this issue

**Proposed Framework:**
- Finetuning LLMs for comparative assessment
- Align model output with target distribution of comparative probabilities
- Training on soft probabilities improves performance and maintains efficiency with a subset of comparisons.

## 1 Introduction

**Introduction:**
- Automatically assessing NLG system quality is challenging
- LLM-as-a-judge: Instruction-tuned LLMs predict text quality of other systems
  * Comparative assessment (Liusie et al., 2024b) demonstrates strong correlation with human judgements, better than absolute assessment (Liu et al., 2023; Liusie et al., 2024a)
- Naive comparative assessment scales quadratically and is impractical for real-world settings
- Efficient comparative assessment using LLM probabilities within a product-of-experts (PoE) framework (Liusie et al., 2024b)

**Beyond Zero-Shot Domain:**
- Benefits gained when systems are fine-tuned for bespoke tasks: absolute assessment (Latif and Zhai, 2024), comparative assessment (Park et al., 2024)
- Experts proposed within PoE framework make strong assumptions about underlying distribution of pairwise probabilities
  * Differences between true and assumed distributions can limit benefits of fine-tuning comparative systems using hard decisions

**Proposed Approach:**
- Address distributional mismatch by training LLM under target distribution
  * Scale pairwise difference in training scores to soft training probabilities under the target distribution
  * Train LLM with these soft pairwise probabilities to match true inference time probabilities and assumed distribution in PoE framework for comparative assessment

**Contributions:**
1. Propose a framework for LLM comparative assessment training
2. Demonstrate that finetuning with soft comparative probabilities under a target distribution enables higher performance with efficient number of comparisons than hard binary decision training.

## 2 Related work

**Comparison of Latent Linear Models (LLMs) for Text Comparisons**
* Recent research on using LLMs for pairwise comparisons and ranking text outputs
* Qin et al. (2024): investigating the use of LLMs in retrieving relevant sources through pairwise comparisons
	+ Full comparison set and sorting-based techniques
* Liu et al. (2024a): computing win-ratio with sets of possible comparisons for medium-sized LLMs
	+ Surpasses traditional scoring methods on NLG assessment benchmarks
	+ Performance declines significantly as the number of comparisons falls
* Liu et al. (2024): limitations of LLM scoring, advocating for pairwise comparisons and introducing PAirwise-preference Search (PAIRS)
	+ Merge sort variant leveraging LLM probabilities
* Liusie et al. (2024b): applying product of experts framework to zero-shot LLM probabilities for higher performing comparative assessment with a subset of comparisons

**Finetuning Prompted Assessment Systems**
* Latif and Zhai (2024): fine-tuning ChatGPT for absolute assessment
* Park et al. (2024) and Ouyang et al. (2022): using human preferences rankings to train reward model under the BradleyTerry model
	+ Hard decisions used in training systems
	+ No consideration of impact on downstream scoring mechanisms, such as the PoE framework.

## 3 LLM comparative assessment

**Comparative Assessment for NLG (Natural Language Generation)**

**Scoring Methods**:
- Objective: Score a set of candidate texts based on an attribute like coherency or question complexity
- Let x1:N denote the set of N candidate texts with their true scores, s1:N
- Let M be a comparative model that returns the probability of xi being greater than xj for the assessed attribute, pij
- Several methods to convert outcomes of pairwise comparisons (C1:K) to predicted scores 	s1:N:
  - **Hard binary decisions**: Win-ratio (Qin et al., 2024; Raina and Gales, 2024) and Bradley-Terry model (Bradley and Terry, 1952)
  - **Probabilities**: Average probability (avg-prob) (Park et al., 2024), PoE-BT in the Framework of Linguistic Models

**Converting Training Scores to Comparative Probabilities**:
- Product of experts perspective assumes a certain distribution to the LLM probabilities
- Zero-shot comparative prompting may not match the assumed distribution
- Finetune LLMs for comparative assessment and control the distribution of returned probabilities
  - Convert training scores to pairwise probabilities using Equation 3: pij = f(si-sj) γσs, where σs is the standard deviation of training scores, γ controls the spread
  - Train the LLM based on soft binary cross entropy loss (Equation 4) with the labeled and predicted probabilities

**Performance Comparison**:
| Model | Mode         | USMLE CMCQRD ρ(↑) r(↑) rmse (↓) | Full [O(N2)] GPT-4 O minizero-shot | Hard binary decisions | Soft binary decisions |
|--------|-------------|----------------------------------|-----------------------|----------------------|---------------------|
|         |            |                                |                       |                    |                   |
| Llama-3.1-8B | Zero-shot  | 22.9, 27.4, 30.4, 12.1, 12.9, 10.07 | 61.3, 57.4, 25.9, 48.1, 49.3, 8.83 | 59.6, 56.4, 26.1, 41.3, 39.1, 9.35 | 61.3, 57.4, 25.9, 48.1, 49.3, 8.83 |
| GPT-4   | Minizero-shot | 27.8, 21.5, 30.9, 14.5, 16.7, 10.00 | 64.8, 60.4, 25.2, 50.9, 52.6, 8.64 | 53.8, 47.3, 30.3, 13.7, 16.7, 10.06 | 60.4, 56.4, 26.1, 40.9, 42.6, 8.64 |
|         |            |                                |                       |                    |                   |

**Note**: The authors also considered Gaussian experts for the PoE framework but focused on soft Bradley-Terry expert as it performed marginally better.

## 4 Experiments

**Data**
- **USMLE**: medical multiple-choice reading comprehension dataset
- **CMCQRD**: educational MCRC dataset
- Comparative study: USMLE vs. CMCQRD (to the author's knowledge, first large datasets with human annotated attributes)
- Dataset statistics
  - USMLE: 667 items, standard split: 466 training, 201 testing, unique contexts
  - CMCQRD: 658 items, no standard split, 78 unique contexts across the whole dataset, no overlap between train and test splits
- USMLE has difficulty scores

**Models**
- **Comparative system**: instruction-tuned LLM (LangModel) with an appropriate prompt for comparative assessment
- For soft finetuning:
  - Calculate probability pij using Equation 4, soft probabilities from Equation 3 for labels
  - Find the best results with γ=5.0
- Comparison of closed source (GPT4o mini) vs open-source solution (Llama-3.1-8B):
  - Llama: zero-shot, hard finetuning and soft finetuning
  - GPT4o mini: only possible to do hard finetuning due to closed source access.

## 5 Results

**Table 1 Summary:**
- Performance metrics: Spearman's correlation coefficient (ρ), Pearson’s correlation coefficient (r), root mean squared error (rmse) between predicted and true scores on each test set using PoE-BT for comparative assessment.
- Hard finetuning GPT4o mini substantially boosts performance compared to zero-shot numbers.
- Similar improvements observed from zero-shot performance when hard finetuning Llama-3.1-8B.
- Figures 1 and 2 present Pearson correlation evolution with efficient number of comparisons; hard finetuning leads to improved performance for a small number of comparisons compared to zero-shot curves.
- Soft finetuning minimally degrades PoE-BT curve with an extremely small subset of comparisons.

**Table 1 Benefits of Soft Finetuning:**
- Selecting high γin soft finetuning pushes distribution of pairwise probabilities outside saturation region of sigmoid.
- Few comparisons needed for each item to deduce overall ranking.

**Comparison with Baselines:**
- Table 3 shows our best comparative system outperforms all submitted solutions (Rodrigo et al., 2024; Tack et al., 2024; Felice and Karaoz, 2024) to BEA shared task 2024 for response time.
- Approach rmse (↓): Dummy Regressor, Baseline, UNED run2, ITEC Lasso, EduTec roberta, Ours: comparative.

## 6 Conclusions

**Proposed Framework for Finetuning LLMs for Comparative Assessment Tasks**
* Propose a framework to fine-tune large language models (LLMs) for comparative assessment tasks
* Aim to achieve the same performance with an efficient subset of comparisons due to quadratic compute cost in full sets
* Fine-tune LLMs using:
  * Binary decisions
  * Soft probabilities calculated from training items' scores using sigmoid function
* Enable pairwise comparisons setup on Bradley-Terry method for near maximal performance with fewer comparisons.

## 7 Limitations

**Limitations of Comparative Assessment:**
- Two different LLMs used: **GPT4o mini** (closed-source) and **Llama-3.1-8B** (open-source)
- Smallest models in their series
- Ideally, larger models would be used for replication experiments
- Lack of computational budget limits the use of larger scale models.

