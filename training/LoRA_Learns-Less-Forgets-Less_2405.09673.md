# LoRA Learns Less and Forgets Less

[2405.09673](https://arxiv.org/abs/2405.09673)
**Authors:** Dan Biderman (1,2), Jacob Portes (2), Jose Javier Gonzalez Ortiz (2), Mansheej Paul (2), Philip Greengard (1), Connor Jennings (2), Daniel King (2), Sam Havens (2), Vitaliy Chiley (2), Jonathan Frankle (2), Cody Blakeney (2), John P. Cunningham (1,3)

**Affiliations:**
- **Columbia University**: Dan Biderman (db3236, pg2118, jpc2181)@columbia.edu
- **Databricks Mosaic Research**: Jacob Portes (jacob.portes), Jose Javier Gonzalez Ortiz (j.gonzalez), Mansheej Paul (mansheej.paul), Connor Jennings (connor.jennings), Daniel King (daniel.king), Sam Havens (sam.havens), Vitaliy Chiley (vitaliy.chiley), Jonathan Frankle (jfrankle), Cody Blakeney (cody.blakeney)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background](#2-background)
- [3 Experimental Setup](#3-experimental-setup)
  - [3.1 Datasets for Continued Pretraining (CPT) and Instruction Finetuning (IFT)](#31-datasets-for-continued-pretraining-cpt-and-instruction-finetuning-ift)
  - [3.2 Measuring Learning with Coding and Math Benchmarks (target domain evaluation)](#32-measuring-learning-with-coding-and-math-benchmarks-target-domain-evaluation)
  - [3.3 Forgetting Metrics (source domain evaluation)](#33-forgetting-metrics-source-domain-evaluation)
- [4 Results](#4-results)
  - [4.1 Target-domain performance: LoRA at low ranks underperforms full finetuning](#41-target-domain-performance-lora-at-low-ranks-underperforms-full-finetuning)
  - [4.2 LoRA forgets less than full finetuning](#42-lora-forgets-less-than-full-finetuning)
  - [4.3 The Learning-Forgetting Tradeoff](#43-the-learning-forgetting-tradeoff)
  - [4.4 For the Tülu-v2-mix dataset, LoRA is on par with full finetuning](#44-for-the-tülu-v2-mix-dataset-lora-is-on-par-with-full-finetuning)
  - [4.5 How strongly does LoRA constrain the finetuning process?](#45-how-strongly-does-lora-constrain-the-finetuning-process)
  - [4.6 Full finetuning on code and math does not learn low-rank perturbations](#46-full-finetuning-on-code-and-math-does-not-learn-low-rank-perturbations)
  - [4.7 Hyperparameter sensitivity analyses for LoRA](#47-hyperparameter-sensitivity-analyses-for-lora)
- [5 Related Work](#5-related-work)
- [6 Discussion](#6-discussion)
- [7 Conclusion](#7-conclusion)
- [Appendix A Experimental Setup](#appendix-a-experimental-setup)
  - [A.1 Training the input and output embedding layers.](#a1-training-the-input-and-output-embedding-layers)
- [Appendix B Learning rate searches](#appendix-b-learning-rate-searches)
  - [B.1 Learning rate sensitivity analysis across optimizers](#b1-learning-rate-sensitivity-analysis-across-optimizers)
  - [B.2 The importance of the alpha scaling parameter for LoRA](#b2-the-importance-of-the-alpha-scaling-parameter-for-lora)
- [Appendix C Finetuning on the Tülu-v2-mix dataset](#appendix-c-finetuning-on-the-tülu-v2-mix-dataset)
  - [C.1 Experimental setup](#c1-experimental-setup)
  - [C.2 Results](#c2-results)
- [Appendix D Supplementary tables](#appendix-d-supplementary-tables)
- [Appendix E Supplementary Figures for SVD Analysis](#appendix-e-supplementary-figures-for-svd-analysis)
- [Appendix F Solution Generation Diversity on HumanEval](#appendix-f-solution-generation-diversity-on-humaneval)
- [Appendix G Training Datasets](#appendix-g-training-datasets)
  - [G.1 MetaMathQA (Math IFT)](#g1-metamathqa-math-ift)
  - [Example G.2: MetaMathQA Problem Solving](#example-g2-metamathqa-problem-solving)
  - [G.2 Magicoder-Evol-Instruct-110k (Code IFT)](#g2-magicoder-evol-instruct-110k-code-ift)
  - [G.3 Starcoder Python (Code CPT)](#g3-starcoder-python-code-cpt)
  - [G.4 OpenWebMath (Math CPT)](#g4-openwebmath-math-cpt)
- [Appendix H Theoretical Memory Efficiency Gains with LoRA for Single and Multi-GPU Settings](#appendix-h-theoretical-memory-efficiency-gains-with-lora-for-single-and-multi-gpu-settings)
  - [H.1 Training on a Single GPU](#h1-training-on-a-single-gpu)
  - [H.2 Training on Multiple GPUs with Fully Sharded Data Parallelism](#h2-training-on-multiple-gpus-with-fully-sharded-data-parallelism)
- [Appendix I LoRA Throughput and Memory Measurements](#appendix-i-lora-throughput-and-memory-measurements)

## Abstract

**Low-Rank Adaptation (LoRA)**:
- Widely-used parameter-efficient finetuning method for large language models
- **Saves memory** by training only low rank perturbations to selected weight matrices

**Evaluation on Target Domains**:
- Comparison of LoRA and full finetuning on two target domains:
  - Programming
  - Mathematics
- Considered two data regimes:
  - **Instruction finetuning**: approximately equal to 100K prompt-response pairs
  - **Continued pretraining**: approximately equal to 20B unstructured tokens

**Findings**:
- In standard low-rank settings, LoRA **substantially underperforms full finetuning**
- However, LoRA better maintains the base model's performance on tasks outside the target domain
- LoRA **mitigates forgetting** more than common regularization techniques like weight decay and dropout
- Helps maintain **more diverse generations**
- Full finetuning learns perturbations with a rank that is 10-100 multiplied by greater than typical LoRA configurations, possibly explaining some of the reported gaps

**Conclusion**:
- Proposed best practices for finetuning with LoRA.

## 1 Introduction

**LoRA vs Full Finetuning:** A Comparative Study on Large Language Models (LLMs)

**Background:**
- LoRA: Low-Rank Adaptation for finetuning large language models with reduced memory footprint
- Debate over performance relative to full finetuning

**Findings:**
1. **Target Domain Performance**: LoRA underperforms full finetuning in continued pretraining (Sec. [4.1](https://arxiv.org/html/2405.09673v2#S4.SS1))
- In instruction finetuning, high ranks can match full finetuning performance (Sec. [4.1](https://arxiv.org/html/2405.09673v2#S4.SS1))
2. **Forgetting Mitigation**: LoRA forgets less than full finetuning and classic regularization techniques (Sec. [4.2](https://arxiv.org/html/2405.09673v2#S4.SS2), Sec. [4.3](https://arxiv.org/html/2405.09673v2#S4.SS3))
3. **Learning-Forgetting Tradeoff**: LoRA helps maintain the diversity of generations (Sec. [4.5](https://arxiv.org/html/2405.09673v2#S4.SS5))
4. **Weight Perturbations**: Full finetuning learns high rank weight perturbations; LoRA learns low-rank perturbations (Sec. [4.6](https://arxiv.org/html/2405.09673v2#S4.SS6))
5. **Hyperparameter Sensitivity**: LoRA is sensitive to learning rates, choice of target modules, ranks, and scaling factors (Sec. [4.7](https://arxiv.org/html/2405.09673v2#S4.SS7))

**Conclusion:**
- Full finetuning is more accurate and sample-efficient than LoRA in continued pretraining for code and math
- In instruction finetuning, higher ranks can close most of the gaps between LoRA and full finetuning performance
- LoRA forgets less than common regularization techniques and maintains diversity of generations
- Full finetuning finds high rank weight perturbations; LoRA learns low-rank perturbations
- A hyperparameter sensitivity analysis for LoRA, along with practical recommendations.

## 2 Background

**LoRA (Layer-wise Relevance Adaptation)**

**Overview**:
- Involves freezing pretrained weight matrix W_{{pretrained}} and learning a low-rank perturbation \Delta
- Reduces memory and FLOPS required for computing the gradient by training fewer parameters

**Implementation**:
- Initialize A_{0}\sim\mathcal{N}(0,1),~{}B_{0}=0
- Set scalar \gamma_{r}=\alpha/r, a controllable hyperparameter
- Choose target modules (W_{{pretrained}} to adapt) and rank (d → r, k)

**Benefits**:
- Trains fewer parameters per module compared to original parameter count
- Example: Applying r=16 LoRA adapter to a 7B weight matrix reduces training by <1%>

**Applications**:
- Initially targeted W_{q} and W_{v} matrices in self-attention modules
- Now best practice to target all transformer modules: \{W_{q}^{(l)}, W_{k}^{(l)}, W_{v}^{(l)}, W_{o}^{(l)}\}_{l=1}^{L} for self-attention, and \{W_{{gate}}^{(l)}, W_{{up}}^{(l)}, W_{{down}}^{(l)}\}_{l=1}^{L} for feedforward modules

## 3 Experimental Setup

### 3.1 Datasets for Continued Pretraining (CPT) and Instruction Finetuning (IFT)

**Datasets Used for Coding and Math Models:**

**1. Coding Dataset: Starcoder-Python (GitHub Repositories)**
   * Permissively licensed repositories from GitHub in Python language
   * Subset of 80+ programming languages, sub-sampled to 20B tokens

**2. Math Dataset: OpenWebMath**
   * Derived from mathematical web pages from Common Crawl
   * Contains LaTeX equations and full English sentences
   * Approx. 14.7B tokens
   * Matched with Starcoder-Python dataset to reach 20B tokens

**3. Coding Dataset: Magicoder-Evol-Instruct-110k (GitHub Repositories)**
   * IFT (inference fine-tuning) dataset for coding tasks

**4. Math Dataset: MetaMathQA**
   * Built by rewriting mathematical word problems using GPT-3.5
   * Contains 395K question-answer pairs and roughly 103M tokens.

**Notes:**
- **Starcoder-Python**: GitHub repositories in Python language, sub-sampled to 20B tokens.
- **OpenWebMath**: Mathematical data from Common Crawl with LaTeX equations and English sentences, matched with Starcoder-Python dataset for 20B tokens.
- **Magicoder-Evol-Instruct-110k**: IFT dataset for coding tasks.
- **MetaMathQA**: Math dataset containing question-answer pairs generated using GPT-3.5 from mathematical word problems in GSM8K and MATH datasets.

### 3.2 Measuring Learning with Coding and Math Benchmarks (target domain evaluation)

**Coding - HumanEval [^6]:**
- Contains 164 coding problems requiring Python program generation from docstring and function signature
- Generated code must pass all supplied unit tests for a correct response
- Code Generation LM Evaluation Harness [^4] with 50 generations per problem, "pass@1" with softmax temperature=0.2 and top\_p=0.95 for 0-shot HumanEval

**Math - GSM8K [^9]:**
- Includes 8.5K grade-school math word problems
- Evaluated on the test split of GSM8K (1,319 samples) using LM Evaluation Harness [^16], default generation parameters (temperature=0, 5 few-shot, pass@1)

### 3.3 Forgetting Metrics (source domain evaluation)

**Benchmarks for Assessing Degradation of Base Model Capabilities:**

**1. HellaSwag [^77]:**
- Includes: 70K problems describing nuanced everyday situations with multiple possible continuations
- Task: Identify the most plausible continuation based on context
- Requires: Commonsense reasoning and understanding of complex social interactions

**2. WinoGrande [^54]:**
- Includes: 44K problems requiring ambiguous pronoun resolution in sentences
- Assesses: Commonsense reasoning
- Multiple choices: Yes

**3. ARC-Challenge [^8]:**
- Includes: 7,787 multiple-choice science questions testing complex reasoning and scientific understanding
- Level: Grade school
- Multiple choices: Yes
- Forgetting metrics: Computed using MosaicML Gauntlet evaluation harness [^11]

**4. Evaluation:**
- Metrics: Calculate accuracy based on predicted logits
- No generation hyperparameters required.

## 4 Results

### 4.1 Target-domain performance: LoRA at low ranks underperforms full finetuning

**LoRA vs. Full Finetuning Comparison**

**Background:**
- Exhaustive learning rate sweep for LoRA and full finetuning
- Importance of ranking in performance comparison

**Results:**

**Code Performance (CPT):**
- IFT datasets show significant gap between full finetuning and LoRA
  - Best LoRA model with rank r=256 peaks at HumanEval=0.224, roughly matching full finetuning with 4B tokens
  - Full finetuning reaches peak HumanEval of 0.263 at 20B tokens
- Ordering by rank emerges after initial 1B CPT tokens
  - For r=16 and r=64, lower accuracy than full finetuning: 0.358 and 0.417 at epoch 4, respectively
  - With high LoRA rank (r=256), full finetuning performance can be matched (LoRA=0.498 in epoch 4)

**Code Solution Generation Diversity:**
- More sensitive HumanEval analysis with pass@ k and temperature 0.8
- Full finetuning superior to r=256 for k < 64, after which they are equal

**Math Performance (CPT):**
- Results closely echo those of code CPT
- Consistent patterns in GSM8K dataset
  - Full finetuning opens a gap, widening with more data
  - Best LoRA (r=256) peaks at 16B tokens (GSM8K=0.203), underperforming full finetuning at 4B tokens (GSM8K=0.224) and peak performance (GSM8K=0.293)

**Math Performance (IFT):**
- LoRA closes much of the gap with full finetuning while remaining less sample efficient
- Both methods substantially improve upon base model
  - LoRA (r=256) peaks at 8 epochs (GSM8K=0.634)
  - Full finetuning achieves GSM8K=0.641 at 2 epochs and peaks at 4 epochs, with GSM8K=0.642

**Conclusion:**
- In CPT, LoRA underperforms full finetuning across all configurations
- In IFT, especially in code, high LoRA ranks are required to close the gap with full finetuning.

### 4.2 LoRA forgets less than full finetuning

**Forgetting and LoRA vs Full Finetuning:**
- **Observations**: IFT induces more forgetting than CPT, programming induces more forgetting than math, forgetting worsens with training duration (Figure 2)
- **LoRA**: forgets less than full finetuning, the extent of forgetting controlled by rank
- **CPT vs LoRA**: Full finetuning scores lower in code than LoRA r=256 (0.545000000000000 vs 0.617000000000000), nearly as much forgetting for IFT with LoRA and full finetuning in math (OpenWebMath has least forgetting)
- **Tradeoff**: LoRA learns less and forgets less relative to full finetuning (Figure 3), each dot represents a separate model with varying training durations.

### 4.3 The Learning-Forgetting Tradeoff

**LoRA vs Full Finetuning: Learning-Forgetting Tradeoffs**

**Trivial vs Nontrivial Question**:
- LoRA models that are finetuned to a new target domain forget less of the source domain compared to full finetuning
- The nontrivial question is: do LoRA and full finetuning differ in how they trade off learning and forgetting?

**Forming Learning-Forgetting Pareto Curves**:
- Plotting the forgetting metric versus the learning metric for each training duration
- Models learn more and forget more as they train on more data
- As LoRA ranks increase, the curves shift up and left, learning more and forgetting more consistently in IFT than CPT

**Dataset Analysis**:
- **Code CPT**: Full finetuning reaches higher HumanEval scores but appears to forget more for any given HumanEval value, while LoRA can reach similar values with less forgetting if trained on more tokens
- **Math CPT**: LoRA and full finetuning curves are roughly overlapping until full finetuning shoots up to achieve much higher GSM8K scores without increased forgetting
- **Code IFT**: LoRA (r=256) offers comparable HumanEval accuracy while strictly forgetting less, with lower ranks not reaching high values of HumanEval
- **Math IFT**: LoRA and full finetuning seem to lie on adjacent learning-forgetting tradeoff curves, with full finetuning offering preferable tradeoffs

**LoRA vs Attention Dropout and Weight Decay**:
- LoRA (r=256) learns as much as full finetuning, weight decay, and attention dropout while forgetting much less (see Figure 4)

### 4.4 For the Tülu-v2-mix dataset, LoRA is on par with full finetuning

**LoRA and Full Finetuning Results on Tülu-v2-mix Dataset**

**Background:**
- LoRA and full finetuning specialized in specific domains: code or math problems within larger IFT data mixtures
- Analysis of their performance on the Tülu-v2-mix dataset [^26] presented in Appendix [C](https://arxiv.org/html/2405.09673v2#A3) and Table [S13](https://arxiv.org/html/2405.09673v2#A4.T13)
- Both LoRA and full finetuning improve upon the base model [^82], [^9], [^23]

**LoRA Performance:**
- Matches full finetuning in chat quality as measured by: Multi-Turn Benchmark (MT-bench), GSM8K, Massive Multitask Language Understanding (MMLU)
- Forgets less at longer training durations (6 epochs)

**Table S13:**
- Detailed supplementary tables for LoRA and full finetuning results on Tülu-v2-mix dataset.

### 4.5 How strongly does LoRA constrain the finetuning process?

**LoRA vs Classic Regularization Techniques**

**Comparison of Learning-Forgetting Tradeoffs**:
- LoRA (r=16,256, training all modules) compared to:
  - **Weight decay** with values 5e^{-5},1e^{-4}
  - **Attention dropout** with values (0.05, 0.1)
- Both regularization techniques learn and forget as much as full finetuning, except:
  - Weight decay starts to generally deteriorate at longer training durations (epochs 8 and 16)
- **LoRA (r=16) learns less and forgets less than all other models
- **LoRA (r=256) learns as much as the other methods while forgetting less

**LoRA Helps Maintain Diversity of Token Generations**:
- Analyzed generated solution strings for **HumanEval problems**
- Calculated unique number of output strings out of 50 generations for:
  - Base model, full finetuning, and LoRA
- Found that **full finetuning** results in fewer unique generations ("distribution collapse") compared to the base model, for both pass and fail generations
  - LoRA was in between the two
- Suggests that LoRA could even substitute a common Kullback-Leibler divergence term that keeps the probabilities of the generated text similar between the finetuned and base model
- **Exact string matching** between generations is not a sensitive metric of predictive diversity, as generations can slightly vary in format and remain functionally identical.

### 4.6 Full finetuning on code and math does not learn low-rank perturbations

**LoRA Training vs Full Finetuning**

**Low-Rank Approximation**:
- Analyzing continued pretraining for code with drastic differences between LoRA and full finetuning
- Focusing on checkpoints obtained at 0.25B, 0.5B, 1B, 2B, 4B, 8B, 16B, and 20B training tokens

**Singular Value Decomposition (SVD) Analysis**:
- Examining the W_q projection at layer 26 of Llama-2-7B (dimensions d×d, d=4096)
- Findings:
  - Spectrum of finetuned weight matrix is similar to base weight matrix, both decaying slowly
  - Difference \Delta also has a similar spectrum to the finetuned and base weight matrices
  - Similar spectra can be achieved by adding low-magnitude Gaussian i.i.d noise to a weight matrix
- Suggests that any transformer model can be well approximated with r=d/2

**Rank of \Delta**:
- Estimating the rank needed to explain 90% of the variance in the matrix
- Findings:
  - Earliest checkpoint at 0.25B CPT tokens exhibits \Delta matrices with a rank 10-100multiplied by larger than typical LoRA ranks
  - Rank of \Delta increases when trained on more data
  - MLP modules have higher ranks compared to attention modules
  - First and last layers seem to be lower rank compared to middle layers

**Dynamic of Rank**:
- Results for Llama-2-7B trained on the Starcoder (CPT) data, showing the rank needed to explain at least 90% of the variance (maximal dimensionality is 4096)

**Targeting MLP or All Modules vs Training Attention Modules Alone**:
- Comparison of LoRA checkpoints trained on Magicoder for 1, 2 and 4 epochs with rank 16, 64, and 256

### 4.7 Hyperparameter sensitivity analyses for LoRA

**LoRA Configuration for Best Performance**

**Goal**: Optimal configuration of LoRA for best chances of matching full finetuning performance.

**Hyperparameters**:
- **Choice of \alpha**: Crucial for high ranks, set to \alpha=2r. Most packages scale matrices by \alpha/r, but scaling down higher ranks may lead to instability and inferior performance.
- **Learning Rates**: Sweep over learning rates with \alpha=512 for best results.
- **Target Modules and Rank**:
  - Training Llama-2-7B models on Magicoder dataset:
    - Performance increases with rank (r=16, 64, 256).
    - Targeting "All" transformer modules (Attention, MLP) drives most gains.
    - MLP blocks are primary loci for continual learning in LoRA.
- **LoRA vs Full Finetuning**:
  - Sensitive to learning rates compared to full finetuning.
  - Benefits from highest stable learning rate within chosen training duration.
  - Best learning rates range between 1e^{-5} and 5e^{-4}.

**Additional Information**:
- Benchmarking throughput and peak GPU memory in Appendix I.
- LoRA tends to train slower than full finetuning with standard implementations and fixed batch size.

**Recommendations**:
- Use LoRA for instruction finetuning, not continued pretraining.
- Target "All" transformer modules (r=256) if GPU memory allows.
- Set \alpha=2r and sweep over learning rates between [1e^{-5}, 5e^{-4}].

## 5 Related Work

**LoRA Extensions:**
- Improvements to LoRA training: initialization [^45], scaling [^74], sequential procedures [^57]
- Alternative low-rank approximations [^38]
- Comparison of techniques left for future work

**Benchmarking LoRA vs. Full Finetuning:**
- Original LoRA paper results: matched full finetuning performance on GLUE, E2E NLG Challenge, WikiSQL [^25]
- Few studies compared LoRA to full finetuning with challenging domains and larger models [^10], [^26], [^84]
- Mixed conclusions regarding practical details: target modules [^52], [^10], rank [^38], [^79]

**Continual Learning on Code and Math:**
- Pretraining LLMs from scratch for code [^36], [^21], [^2]
- Combining continued pretraining and instruction finetuning (IFT) on large datasets [^6], [^3], [^53]
- IFT only: MagiCoder [^43], WizardCoder [^43]
- LoRA used for IFT: OctoCoder [^46]
- Improving mathematical capabilities: DeepSeek Math [^56], auto-supervised methods [^41], Monte Carlo Tree Search [^76], Code-Interpreter-style solutions [^58]
- Mitigating forgetting: prompt tuning [^34], replaying source-domain data [^33].

## 6 Discussion

**LoRA vs Full Finetuning: Differences and Limitations**

**Effectiveness of Finetuning and Model Size**:
- Past studies suggest a relationship between finetuning effectiveness and model size
- LoRA has been successfully applied to 70B parameter models [^26]
- Techniques like prompt tuning become more effective for larger models [^67]
- Rigorous study of these scaling properties left for future work

**Limitations of Spectral Analysis**:
- Observation that full finetuning tends to find high rank solutions does not rule out low rank solutions
- Possibility of a higher rank needed to reconstruct the weight matrix for downstream tasks
- SVD analysis presented only for continued pretraining setting
- Different interpretation for instruction finetuning setting may reveal lower ranks for full finetuning.

## 7 Conclusion

**Study Findings on LoRA and Full Finetuning Performance:**
- **Downstream performance of 7 billion parameter LLMs**: results from domain-specific datasets in code and math with sensitive evaluation metrics
- **LoRA underperforms full finetuning**: across various domains using commonly used low-rank settings
- **Behavior preservation**: LoRA keeps finetuned model's behavior close to base model, reducing source-domain forgetting and promoting diverse generations at inference time
- **Forgetting mitigation**: LoRA outperforms classical regularization techniques in addressing forgetting
- **Weight perturbations**: full finetuning finds weight perturbations that deviate significantly from being low-rank

**Acknowledgements:**
- Gratitude to editor and anonymous reviewers for valuable feedback
- Recognition of Daniel Han, Damjan Kalajdzievski for their contributions in reading the work and highlighting importance of setting \alpha=2r during high rank training.

**Author Contributions:**
- **D.B.**: project leader, code development, experiment running, result analysis, manuscript writing
- **J.P.**: experiment running, manuscript assistance
- **J.G.O.**: code writing and experiment execution
- **P.G.**: advised SVD analysis
- **C.J.**: ran experiments
- **D.K.**: code writing
- **M.P., S.H., V.C., J.F., C.B., J.P.C.**: advised the work.

## Appendix A Experimental Setup

**LoRA Configuration for All Experiments**

**Hardware and Libraries:**
- Experiments done using Databricks MosaicML composer, streaming and llm-foundry libraries, and HuggingFace peft library on 32× H100-80GB GPUs.

**Optimizer:**
- LionW optimizer used instead of AdamW for all experiments.

**Targeted Modules:**
- All trainable modules inside each Llama transformer block: \{W\_q^{(l)}, W\_k^{(l)}, W\_v^{(l)}, W\_o^{(l)}, W\_{gate}^{(l)}, W\_{up}^{(l)}, W\_{down}^{(l)}\}_{l=1}^{L}

**Rank and Scaling Factor:**
- Ranks: r=16, 64, 256
- Scaling factor \gamma\_r=2 across ranks
- lora\_dropout=0.05

**Training:**
- All models trained on 20B tokens
- Individual cooldowns using intermediate checkpoints:
  - Target max training duration (e.g., 8 billion tokens)
  - Last 20% of max training duration as the cooldown period
  - Retrain from latest available checkpoint prior to the cooldown period

**Code CPT:**
- Llama-2-7B trained on StarCoder-Python dataset
  * Optimizer: decoupled\_lionw (betas=[0.9, 0.95])
  * Learning rate: 1.0e-05 for LoRA and Full Finetuning
  * Scheduler: inv\_sqrt\_with\_warmup (t\_scale=1000ba, t\_warmup=1000ba, t\_cooldown=5086ba, alpha\_f\_decay=1, alpha\_f\_cooldown=0)
  * Global train batch size: 192
  * Device train microbatch size: 6
  * Gradient clipping: norm (threshold=1)
- Math CPT: Llama-2-7B trained on OpenWebMath dataset
  * Optimizer: decoupled\_lionw (betas=[0.9, 0.95])
  * Learning rate: 1.0e-05 for full finetuning, 4.0e-05 for LoRA
  * Scheduler: inv\_sqrt\_with\_warmup (t\_scale=1000ba, t\_warmup=1000ba, t\_cooldown=5086ba, alpha\_f\_decay=1, alpha\_f\_cooldown=0)
  * Global train batch size: 192
  * Device train microbatch size: 6
  * Gradient clipping: norm (threshold=1)

**Code IFT:**
- Finetuning Llama-2-7B on Magicoder-Evol-Instruct-110K dataset
  * Optimizer: decoupled\_lionw (betas=[0.9, 0.95])
  * Learning rate: 2e-4 for r=16,64, 1e-4 for r=256 (\alpha=2r=512) due to instabilities/loss spikes at 2e-4
  * Scheduler: cosine\_with\_warmup (alpha\_f=0.01, t\_warmup=0.1dur)
  * Global train batch size: 192
  * Device train microbatch size: 6
  * Gradient clipping: norm (threshold=1)

**Math IFT:**
- Finetuning Llama-2-7B on MetaMathQA dataset
  * Optimizer: decoupled\_lionw (betas=[0.9, 0.95])
  * Learning rate: Full finetuning: 1e-5, LoRA: 1e-4 for r=16,64, 5e-5 for r=256 due to instabilities.
  * Scheduler: cosine\_with\_warmup (alpha\_f=0.01, t\_warmup=0.1dur)
  * Global train batch size: 768
  * Device train microbatch size: 24
  * Gradient clipping: norm (threshold=1)

### A.1 Training the input and output embedding layers.

**LoRA vs Full Finetuning**
- **Vanilla LoRA**: does not train input and output embedding layers
- **QLoRA**: same as Vanilla LoRA
- Recent open-source work: shows benefits of supplementing LoRA with full finetuning (additional ≈200M parameters)
- Approach is a hybrid of LoRA and full finetuning: leaves empirical investigation for future work
- Involves hyperparameter optimization: separate learning rates for input, output layers (smaller than LoRA learning rates)

**Fine-tuning**:
- Recent open-source work shows benefits of full finetuning with LoRA
- Hybrid approach involves further hyperparameter optimization: separate learning rates for input and output layers

## Appendix B Learning rate searches

**LoRA vs Full Finetuning: Learning Rate Sensitivity Analysis**

**Findings**:
- LoRA improves monotonically with learning rate up to a value at which training diverges
- Best LoRA learning rates for code and math are 5e^{-4} and 2e^{-4}, respectively
- These best LoRA learning rates are underperformed by four alternative full finetuning learning rates:
    - Code: 5e^{-5} and 1e^{-5} (order of magnitude smaller than LoRA)
    - Math: No viable alternative learning rates identified
- For LoRA, the target modules exclude W_{gate}, with \alpha=32 (should be higher for r=64)

**Comparison to Full Finetuning**:
- Differences in best learning rates between LoRA and full finetuning
- LoRA is more sensitive to learning rates compared to full finetuning

**Experiments**:
- Llama-2-7B models trained on:
    - **Code**: Magicoder-Evol-Instruct-110k
    - **Math**: MetaMathQA
- Evaluated on:
    - HumanEval (code)
    - GSM8K (math)
- Experiments performed with LionW; comparison to AdamW in Figure S2.

### B.1 Learning rate sensitivity analysis across optimizers

**Comparison of AdamW and Decoupled LionW Optimizers**

Training Magicoder-Evol-Instruct-110K for two epochs using various learning rates revealed that Decoupled LionW outperformed AdamW in HumanEval for LoRA and full finetuning. Across different learning rates, Decoupled LionW showed superiority as shown in Fig. [S2](https://arxiv.org/html/2405.09673v2#A2.F2 "Figure S2 ‣ B.1 Learning rate sensitivity analysis across optimizers ‣ Appendix B Learning rate searches ‣ LoRA Learns Less and Forgets Less"). [Refer to figure](https://arxiv.org/html/2405.09673v2/extracted/5869312/figures/adam_vs_lion.png) (Figure S2: Comparison of LionW vs AdamW across learning rates for two epochs on the Magicoder-Evol-Instruct-110K dataset, with results in HumanEval and MosaicML evaluation gauntlet's "Language Understanding" benchmarks). Both methods peaked at the learning rate used in the original paper [^71].

### B.2 The importance of the alpha scaling parameter for LoRA

**Performance of LoRA**: High sensitivity to the LoRA \alpha hyperparameter was observed in all models. Figure S3 from [2] demonstrates experiments on two datasets (Magicoder-Evol-Instruct-110K and OpenWebMath) with rank r=256. In both cases, the best accuracy was achieved when \alpha=2r (Figure S3, blue).

[2] <https://arxiv.org/html/2405.09673v2>

![Optimal LoRA settings](https://arxiv.org/html/2405.09673v2/extracted/5869312/figures/magicoder-gate_proj-alpha-sweep-2024-8-1128.png)

(a) (a) Optimal LoRA settings: \alpha=2r (blue), learned jointly with learning rate.

## Appendix C Finetuning on the Tülu-v2-mix dataset

**Finetuning Llama-2-7B on Tülu-v2-mix Dataset:**
- **Finetuned Llama-2-7B models** on Tülu-v2-mix dataset for chain of thought reasoning, assistant conversations, math and science problems, code, etc. (326k samples) [^16]
- Evaluated after 2, 4, and 6 epochs in each of four experimental conditions: full finetuning vs LoRA (r=16, 64, 256) targeting all transformer modules
- No cooling down for checkpoints during training durations
- **Assessment:** Can LoRA with a low rank achieve full finetuning accuracy in specific domains and general conversational capabilities?

**Evaluation Metrics:**
- Math capabilities: GSM8K [^9]
- STEM, humanities, social science: Average of 57 subjects from Massive Multitask Language Understanding (MMLU) [^23]
- Conversational capabilities: Multi-Turn Benchmark (MT-bench) [^82] with 80 multi-turn conversations evaluated automatically by GPT-4
- Average forgetting score for all datasets in the paper.

### C.1 Experimental setup

**Hyperparameters Selected After Initial Learning Rate Sweep:**
- Optimizer: decoupled\_lionw (betas=[0.9, 0.95])
- Learning rate for full finetuning: 5e-6; LoRA: 1e-4
- Scheduler: cosine\_with\_warmup (alpha\_f=0.01, t\_warmup=0.1dur)
- Training batch size: global\_train\_batch\_size = 192
- Microbatch size during training: device\_train\_microbatch\_size = 6
- Gradient clipping: norm (threshold=1)

### C.2 Results

**LoRA vs Full Finetuning Results on Tülu-v2-mix Dataset**

**MT-bench (Figure S4)**
- Both LoRA and full finetuning improve upon base model (2.74) starting from second epoch
- All LoRA models within one standard error of full finetuning mean
- 80 questions in benchmark, high variance

**GSM8K (Figure 5a)**
- All models significantly improve upon base model (0.145)
- LoRA and full finetuning are overlapping: best model is LoRA r=256 at epoch 4
- Full finetuning forgets more than LoRA after epoch 4

**MMLU (Figure 5b)**
- Full finetuning and LoRA are overlapping: best model is LoRA r=64 at epoch 4
- No ordering by rank in this dataset
- Full forgetting exhibits the most forgetting, followed by LoRA ranking

**Comparing Models**
- At two epochs, full finetuning outperforms LoRA
- Afterwards, LoRA needs 4 epochs to train and underperforms in domain-specific knowledge like math
- Future work needed to understand why LoRA underperforms in specific domains

**MT-bench Results (Figure S4)**
- Average MT-bench score with GPT-4 as judge for 80 questions with two turns each
- Base model value reported in the MT-bench paper
- Tülu paper reports a slightly exceeding standard error from average score at epoch 2

**GSM8K Results (Figure S5a)**
- Accuracy in GSM8K dataset for both LoRA and full finetuning models

**Forgetting Results (Figure S6)**
- LoRA forgets less even on a more diverse IFT dataset like Tülu-v2-mix
- Plot of average forgetting score as a function of training duration.

## Appendix D Supplementary tables

**Starcoder-Python Results (HumanEval pass@1, temperature 0.2)**
- Table S1: Starcoder-Python results for HumanEval pass@1 and temperature 0.2

**Table S2: Starcoder-Python Results (Forgetting Average)**
- Table S2: Starcoder-Python results for forgetting average

**OpenWebMath Results (GSM8K)**
- Table S3: OpenWebMath results for GSM8K

**OpenWebMath Results (Forgetting Average)**
- Table S4: OpenWebMath results for forgetting average

**Magicoder-Evol-Instruct-110K Results (HumanEval pass@1)**
- Table S5: Magicoder-Evol-Instruct-110K results for HumanEval pass@1

**Magicoder-Evol-Instruct-110K Results (Forgetting Average)**
- Table S6: Magicoder-Evol-Instruct-110K results for forgetting average

**MetaMathQA Results (GSM8K)**
- Table S7: MetaMathQA results for GSM8K

**MetaMathQA Results (Forgetting Average)**
- Table S8: MetaMathQA results for forgetting average

**Tülu-v2-mix Results**
- **Table S9:** Tülu-v2-mix results

**MMLU (Tülu-v2-mix)**
- **Epoch**: 2, 4, 6
- **Condition**: LoRA (r=16), LoRA (r=64), LoRA (r=256), Full Finetuning
- **LoRA (r=16)**: 5.681, 5.997, 5.712
- **LoRA (r=64)**: 5.597, 5.725, 5.944
- **LoRA (r=256)**: 5.788, 5.834, 5.894
- **Full Finetuning**: 5.825, 5.838, 5.862

**GSM8K (Tülu-v2-mix)**
- **Epoch**: 2, 4, 6
- **Condition**: LoRA (r=16), LoRA (r=64), LoRA (r=256), Full Finetuning
- **LoRA (r=16)**: 0.251, 0.275, 0.280
- **LoRA (r=64)**: 0.285, 0.270, 0.295
- **LoRA (r=256)**: 0.296, 0.335, 0.301
- **Full Finetuning**: 0.324, 0.291, 0.303

**Forgetting Average (Tülu-v2-mix)**
- **Epoch**: 2, 4, 6
- **Condition**: LoRA (r=16), LoRA (r=64), LoRA (r=256), Full Finetuning
- **epoch**: 2, 4, 6
- **condition**: LoRA (r=16), LoRA (r=64), LoRA (r=256), Full Finetuning
- **LoRA (r=16)**: 0.650, 0.657, 0.657
- **LoRA (r=64)**: 0.649, 0.655, 0.647
- **LoRA (r=256)**: 0.653, 0.649, 0.629
- **Full Finetuning**: 0.660, 0.652, 0.621

## Appendix E Supplementary Figures for SVD Analysis

**SVD Analysis of Layer 26 Matrix W\_{q}**
- [Figure S7](https://arxiv.org/html/2405.09673v2/extracted/5869312/figures/single_layer_spectrum_starcoder_cropped.png) demonstrates the Singular Value Decomposition (SVD) for a 4096\multiplied by 4096 matrix at layer 26 of an unspecified model.
- The singular values for base weights, finetuned weights, and their difference are shown on the left. A rank >1500 is required to explain 90% of the variance in all three matrices.
- [Figure (a) (a)](https://arxiv.org/html/2405.09673v2/x1.png) shows the spectrum for matrices A, A+cB, and cB when c=0.1. All three matrices have high rank.

## Appendix F Solution Generation Diversity on HumanEval

**Llama-2-7B Models and HumanEval Benchmark**

**Pass@k Metric**:
- Measured for Llama-2-7B models trained on Magicoder dataset
- Controls the acceptance criterion with parameter k
- Defined as: k = 1 - \frac{{\binom{n-c}{k}}}{\binom{n}{k}}, where n is number of generations, c is number of correct generations, and k determines sample set size
- Increasing k increases diversity of generated generations and likelihood of passing generation in a random subset

**Results**:
- Pass@k scores for LoRA models and base Llama-2-7B model (Figure S9)
- All models consistently improve pass@k as k increases
- Finetuned models outperform base model at all values of k
- Full finetuning is superior across all k with temperature 0.8
- Gap between LoRA and full finetuning reduces when k > 16

**Conclusion**:
- Llama-2-7B models trained on Magicoder dataset show improvement in pass@k scores as k increases
- Finetuned models outperform base model for all values of k with temperature 0.8
- LoRA models learn less and forget less, but still underperform full finetuning at low ranks.

## Appendix G Training Datasets

### G.1 MetaMathQA (Math IFT)

**MetaMathQA Samples**
- **Methods Used:**
  - Answer Augmentation (155k samples) [^75]
  - Rephrasing (130k samples) [^75]
  - Self-Verification [^72]
  - FOBAR [^28]
- **Description of Methods:**
  - **Answer Augmentation**: generates multiple reasoning paths for a mathematical question and filters for correct final answers.
  - **Rephrasing**: uses GPT-3.5 to rephrase questions, checks for correctness by comparing reasoning chains and proposed answers with ground truth answers.
  - **Self-Verification** and **FOBAR**: fall under "backward reasoning," where the question starts with a given condition and requires solving for an unknown variable.
- **Example:**
  - Type: GSM_Rephrased
  - Query: Scott wants to prepare and freeze a large portion of ratatouille... If this yields 4 quarts, how much does each quart cost?
  - Original Question: Scott wants to make and freeze a large batch of ratatouille... If this yields 4 quarts, how much does each quart cost?
  - Response: The cost of each quart is 10.00.

### Example G.2: MetaMathQA Problem Solving

**Example G.2: MetaMathQA**
- **Type**: MATH\_SV
- **Query**:
  - If a school bus leaves school with X students on board, and one-half of the students get off the bus at each of the first three stops,
  - The number of students remaining on the bus after the third stop is 6.
  - What is the value of unknown variable X?

**Original Question**:
- If a school bus leaves school with **48 students** on board, and one-half of the students get off the bus at each of the first three stops,
- How many students remain on the bus after the third stop?

**Solution**:
- To solve this problem, we need to determine the value of x, which represents the initial number of students on the bus.
- Let's break down the information given:
  - Number of students remaining on the bus after the third stop: **6**
  - Number of students getting off the bus at each stop: **1/2**
  - Number of stops: **3**
- We can set up the equation as follows:
  - Number of students on the bus * (1/2)^3 = Number of students remaining on the bus after the third stop
  - x * (1/2)^3 = 6
- To isolate x, we multiply both sides by (1/8):
  - x * (1/8) * 8 = 6 * 8
  - x = **48**
- The answer is: **48**.

### G.2 Magicoder-Evol-Instruct-110k (Code IFT)

**Magicoder-Evol-Instruct-110k Dataset**

**Overview**:
- Contains 72.97M tokens of programming questions and answers
- Reproduces the "Evol-Instruct" dataset of WizardCoder by:
  - Iteratively prompting an LLM (GPT-4) to increase the difficulty of a set of question-answer pairs from Code Alpaca

**Accessing the Dataset**:
- Can be found at: [https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K]

**Example Question and Answer Pair**:
1. **Instruction**: Amend the subsequent Python script so that it includes a 'while' loop instead of the existing 'for' loop, which iterates through the items of an integer list. The script currently has a bug where it attempts to print an object that is outside the bounds of the list. Fix this error and modify the script to use 'while' instead of 'for' loop. Ensure your script correctly handles empty lists.
2. **Response**: Initialize the index at 0 (i = 0), traverse the list using a 'while' loop (while i < n:), output each individual element (print(arr[i])), and increment the index (i += 1). This will iterate over the list until it reaches the end, handling empty lists since the loop will not proceed if the list is empty.

### G.3 Starcoder Python (Code CPT)

**GitHub Dataset**
- Consists of permissively licensed repositories
- Includes Git commits in over 80 programming languages
- Python subset used, sub-sampled to 20B tokens
- Full dataset: [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)

**Starcoder-Python Example**
- Includes code snippet for invoking Gitee API using Python
- Imports required modules: base64, logging, requests
- Defines functions to handle GET requests with URL and params
  - Logs request information
  - Attempts connection and returns response on success
  - Returns error message on failure (ConnectionError or IOError)

### G.4 OpenWebMath (Math CPT)

**OpenWebMath Dataset**
- Contains 14.7B tokens derived from mathematical web pages
- Correctly formatted to preserve mathematical content (e.g., LaTeX equations)
- **Example**:
  - URL: http://math.stackexchange.com/questions/222974/probability-of-getting-2-aces-2-kings-and-1-queen-in-a-five-card-poker-hand-pa
  - Text: "Probability of getting 2 Aces, 2 Kings and 1 Queen in a five card poker hand (Part II) So I reworked my formula in method 1 after getting help with my original question - Probability of getting 2 Aces, 2 Kings and 1 Queen in a five card poker hand. But I am still getting results that differ…although they are much much closer than before, but I must still be making a mistake somewhere in method 1."

**LoRA (Layer-wise Relevance Analysis)**
- Theoretical memory efficiency gains:
  - LoRA learns less and forgets less
  - Enables training on fewer GPUs (multi-GPU setting)
- Practical memory savings:
  - LoRA leads to memory savings relative to full finetuning
  - Can lead to slower throughput for particular hardware and software settings

**Appendix H: Theoretical Memory Efficiency Gains with LoRA**
- Discusses theoretical benefits of LoRA in the single and multi-GPU settings

**Appendix I: LoRA Throughput and Memory Measurements**
- Shows how LoRA leads to memory savings relative to full finetuning in practice
- Investigates potential slower throughput for certain hardware and software configurations.

## Appendix H Theoretical Memory Efficiency Gains with LoRA for Single and Multi-GPU Settings

**Modern Neural Network Training Systems**

**Memory Requirements**:
- **Model states**:
  - Higher order optimization quantities: optimizer momentum and variance (Adam), momentum (Lion)
- **Residual states**:
  - Activations (batch size, sequence length)
  - Temporary buffers for intermediate forward/backward pass quantities

**Memory Savings with LoRA**:
- LoRA offers memory savings in:
  - Single GPU setting
  - Multi-GPU setting
- Examples inspired by [^51]:

**Single Precision Data**:
- Master copy of tuned parameter weights
- All optimizer states (Adam: momentum and variance; Lion: just momentum)
- **Mixed-precision training**:
  - Critical data stored at single precision (fp32)
  - Some computations performed at half precision (fp16 or bfloat16)

### H.1 Training on a Single GPU

**Memory Requirements Comparison: LoRA vs Full Finetuning (Single GPU)**

**Adam Optimizer:**
- **Full finetuning**: Stores master weights in fp32 (4 bytes/param) + gradients (4 bytes/tuned param) + optimizer state (8 bytes/tuned param) = 112 GB for a 7B model
- **Lion Optimizer**: Calculates gradients using momentum term only, eliminating variance term = 84 GB for a 7B model

**LoRA:**
- Does not calculate or maintain optimizer states (momentum and variance terms) for most parameters
- Reduced memory usage:2×\Psi+16×\Psi\times 0.01=$ 15.12 GB with bfloat16 storage (assuming non-tuned param weights are stored in bfloat16)

**Memory Requirements Breakdown:**
- Master weights: fp32 (4 bytes/param)
- Gradients: fp32 (4 bytes/tuned param)
- Optimizer state: momentum term (4 bytes/tuned param), variance term (none for LoRA)

**Total Memory Usage:**
- Full finetuning with Adam: 112 GB for a 7B model
- Lion optimizer full finetuning: 84 GB for a 7B model
- LoRA setting with Adam: 15.12 GB (assuming non-tuned param weights are stored in bfloat16)

**Note**: These calculations do not account for batch size or sequence length effects on activation memory requirements.

### H.2 Training on Multiple GPUs with Fully Sharded Data Parallelism

**Approaches for Training Language Models Across Multiple GPUs:**
- **Model parallelism**: Different layers stored on different GPUs (high communication overhead, poor throughput)
- **Fully Sharded Data Parallelism (FSDP)**: Shards parameters, gradients, and optimizer states across GPUs (efficient, competitive with LoRA in certain settings)

**Memory Usage Comparison:**
*7B parameter model:*
  - With FSDP: total memory requirement per GPU = 3.5 GB (with 32 GPUs)
  - With Adam + LoRA: total memory requirement per GPU = 0.4725 GB (with 32 GPUs)
*Industry level GPUs:*
  - V100s: 16 GB
  - A100s and H100s: 80 GB

**Memory Efficiency Gains with LoRA:**
- For single GPU: 15.12 GB vs. 8 GPUs: 14 GB (no efficiency gains)
- For large models, like 70B or 405B, LoRA is beneficial as the model size scales

**Table S14:** Theoretical memory required to store the model and optimizer state during training for a 7B parameter model. FSDP sharding the parameter and optimizer states across N devices results in less memory usage relative to LoRA. LoRA enables training on GPUs with far less memory and without needing as many GPUs to shard across.

**Table S15:** Theoretical memory required to store the model and optimizer state during training for a 70B parameter model.

**Table S16:** Theoretical memory required to store the model and optimizer state during training for a 405B parameter model.

## Appendix I LoRA Throughput and Memory Measurements

**LoRA vs. Full Finetuning: Training Efficiency Comparison**

**Measured Metrics**:
- Throughput (tokens per second)
- Peak active memory (GB)

**Experimental Setup**:
- Single node of 8x H100-80GB GPUs
- Per-GPU micro batch size: 1
- Targeted all linear layer weights with LoRA (Attention and MLP)

**Observations**:
- Significant gap between full finetuning and LoRA runs due to additional overheads
- Approximately 15% reduction in throughput for given batch size with LoRA
- Higher LoRA ranks are slower than lower ranks across all batch sizes, especially noticeable for rank 512
- Slightly higher throughput for LoRA settings with larger batch sizes
- Slowdown related to more computations of intermediate activations
- Optimizations possible to reduce the gap in throughput using HuggingFace peft library

**Peak Memory**:
- Substantial reduction (~40%) for small batch sizes when using parameter efficient methods
- Increase in memory requirements for intermediate activations with larger batch sizes
- Limit per GPU micro batch size to 8 to prevent out of memory errors
- Throughput and memory stabilize for batch sizes 64 and above, with around 15% memory savings for larger batch sizes.

