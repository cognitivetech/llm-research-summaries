# BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models

**Authors**: Yupeng Chang (1), Yi Chang (1,2,3), Yuan Wu (1,2) (School of Artificial Intelligence, Jilin University), Key Laboratory of Symbolic Computation and Knowledge Engineering, Jilin University, International Center of Future Science, Jilin University

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Works](#2-related-works)
- [3 Method](#3-method)
  - [3.1 Principal Singular Values and Singular Vectors Adaptation (PiSSA)](#31-principal-singular-values-and-singular-vectors-adaptation-pissa)
  - [3.2 Bias-Alleviating Low-Rank Adaptation (BA-LoRA)](#32-bias-alleviating-low-rank-adaptation-ba-lora)
- [4 Experiments](#4-experiments)
  - [4.1 Models and Datasets](#41-models-and-datasets)
  - [4.2 Baselines](#42-baselines)
  - [4.3 Implementation Details](#43-implementation-details)
  - [4.4 Results and Analysis](#44-results-and-analysis)
- [5 Conclusion](#5-conclusion)
- [6 Ethics Statement](#6-ethics-statement)
- [7 Reproducibility](#7-reproducibility)
- [Appendix A Background](#appendix-a-background)
  - [A.1 Challenges of Bias and Noise in Pre-training Data](#a1-challenges-of-bias-and-noise-in-pre-training-data)
  - [A.2 Mitigating Bias and Noise through Parameter-Efficient Fine-Tuning Methods](#a2-mitigating-bias-and-noise-through-parameter-efficient-fine-tuning-methods)
- [Appendix B Details of Models and Datasets](#appendix-b-details-of-models-and-datasets)
  - [B.1 Details of Models](#b1-details-of-models)
  - [B.2 Details of Datasets](#b2-details-of-datasets)
  - [B.3 Specific Hyperparameter Settings of RoBERTa-large and DeBERTa-v3-base on GLUE](#b3-specific-hyperparameter-settings-of-roberta-large-and-deberta-v3-base-on-glue)
  - [B.4 Specific Hyperparameter Settings of BERT-L and GPT-2-XL on GLUE and GLUE-x](#b4-specific-hyperparameter-settings-of-bert-l-and-gpt-2-xl-on-glue-and-glue-x)
- [Appendix C More Experiments](#appendix-c-more-experiments)
  - [C.1 Analysis on Different Sizes and Types of Models](#c1-analysis-on-different-sizes-and-types-of-models)
  - [C.2 Evaluating the Performance of Different Ranks](#c2-evaluating-the-performance-of-different-ranks)
  - [C.3 t-SNE Visualizations of Feature Evolution during the Fine-tuning with LoRA and BA-LoRA](#c3-t-sne-visualizations-of-feature-evolution-during-the-fine-tuning-with-lora-and-ba-lora)
- [Appendix D More Discussions](#appendix-d-more-discussions)
  - [D.1 Limitations](#d1-limitations)
  - [D.2 Future works](#d2-future-works)


## Abstract

**Large Language Models (LLMs)**
- Demonstrated remarkable proficiency across various Natural Language Processing (NLP) tasks
- Adapting LLMs to downstream applications requires:
  - Computationally intensive and memory-demanding fine-tuning procedures

**Parameter-Efficient Fine-Tuning (PEFT)**
- Promising approach to tailor LLMs with minimal computational overhead
- Offers substantial advantages, but does not fully address the issue of **bias propagation from pre-training data**

**Bias-Alleviating Low-Rank Adaptation (BA-LoRA)**
- Novel PEFT method designed to counteract bias inheritance
- Incorporates three distinct regularization terms:
  1. **Consistency regularizer**: Enhances model's consistency during fine-tuning
  2. **Diversity regularizer**: Promotes model diversity and avoids overfitting
  3. **Singular Value Decomposition regularizer**: Improves generalization capabilities

**Experimental Results**
- Conducted on Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks
- Used prominent LLMs such as LLaMA, Mistral, and Gemma
- Demonstrated that BA-LoRA:
  - Outperforms LoRA and its state-of-the-art variants
  - Effectively mitigates the adverse effects of pre-training bias
  - Leads to more reliable and robust model outputs

**Availability**
- The code is available at [https://github.com/cyp-jlu-ai/BA-LoRA](https://github.com/cyp-jlu-ai/BA-LoRA)

## 1 Introduction

**Emergence of Large Language Models (LLMs)**
* GPT-4, Llama, Mistral, Gemma: exceptional performance across NLP tasks
* Demonstrated language comprehension, generation, and reasoning [^84]
* Training on vast datasets [^84]
* Challenges with increasing data volume [^60]
	+ Imbalanced, duplicated, corrupted information
	+ Bias in training data affects model behavior [^22]
		- Noise degrades model generalization [^12]
		- Overemphasis on overrepresented topics [^86]
* Biases persist even after fine-tuning: "Catastrophic Inheritance" [^12]

**Fine-Tuning LLMs**
* Powerful method for enhancing task-specific performance [^29]
* Aligning models with user intent [^59]
* Eliciting desired behaviors [^5]
* Substantial computational and memory demands [^34]
	+ 16-bit fine-tuning of Llama-65B requires over 780 GB GPU memory [^18]

**Parameter-Efficient Fine-Tuning (PEFT)**
* LoRA: low-rank adapter matrices to reduce training overhead [^34]
	+ $A$ and $B$ learnable matrices initialized as normal distribution and zero, respectively
	+ Preserves original output of pre-trained weight matrix $W$
	+ Reduces computational demands compared to full fine-tuning

**Proposed Approach: Bias-Alleviating Low-Rank Adaptation (BA-LoRA)**
* Building upon Principal Singular Values and Singular Vectors Adaptation (PiSSA) [^56]
* Incorporates three distinct regularization terms: consistency, diversity, and SVD regularizer
	+ Consistency preserves valuable pre-trained knowledge during fine-tuning
	+ Diversity encourages varied model outputs
	+ Enhances generalization capabilities of generative models
* Tailored strategies for NLU and NLG

**Evaluation of BA-LoRA**
* Comprehensive experiments across diverse benchmarks: GSM8K [^16], MATH [^80], HumanEval [^14], MBPP [^3], GLUE [^74], MT-Bench [^85]
* Prominent LLMs and encoder-only architectures used for evaluation.

## 2 Related Works

**Parameter-efficient fine-tuning (PEFT)**

**Background:**
- PEFT techniques: adapting LLMs for specific tasks under limited resources
- Three main categories: adapter-based methods, soft prompt tuning, and low-rank adaptation (LoRA)

**Adapter-based methods:**
- Introduce additional layers with fewer parameters
- Fine-tune these layers to reduce computational costs

**Soft prompt tuning methods:**
- Prepend learnable soft prompts to model's input for specific tasks
- Leverage pre-trained models' capabilities, requiring only appropriate prompts

**Low-rank adaptation (LoRA):**
- Introduces low-rank matrices within existing layers
- Approximates weight updates during fine-tuning

**LoRA Variants:**
- **AdaLoRA**: Distributes parameter budget among weight matrices based on importance
- **DoRA**: Decomposes pre-trained weights into magnitude and direction components for fine-tuning
- **LoHA**: Employs Hamiltonian products to enhance LoRA efficiency
- **DyLoRA**: Dynamically trains LoRA blocks across varying ranks
- **DeltaLoRA**: Updates model's original weights using parameters from adapter layers
- **PiSSA**: Initializes adapter matrices to approximate the original matrix through singular value decomposition

**BA-LoRA:**
- Uniquely addresses core challenge of Catastrophic Inheritance in LLM fine-tuning.

## 3 Method

### 3.1 Principal Singular Values and Singular Vectors Adaptation (PiSSA)

**PiSSA: An Adaptation of LoRA for Faster Convergence**

**Overview**: PiSSA is a variant of LoRA that addresses convergence speed challenges by retaining the core LoRA architecture and innovating in initialization. It uses principal components of the original weight matrix to initialize adapter matrices.

**Initialization**:
- Leverages singular value decomposition (SVD) of $W$: $W=USV^{T}$
- Partitions singular values and vectors into principal and residual components
- Initializes low-rank adapters with principal components: $A=U_{[:,:r]}S_{[:r,:r]}^{1/2}$ and $B=S_{[:r,:r]}^{1/2}V_{[:,:r]}^T$
- Freezes the residual matrix during fine-tuning: $W^{res}=U_{[:,r:]}S_{[r:,r:]}V_{[:,r:]}^T$

**Preservation of Full Capacity**:
- Fine-tuning starts with: $W=W^{res}+AB$
- Prioritizes training the most influential parameters for faster convergence

**Benefits**:
- Inherits LoRA's benefits of reduced parameter count and deployment simplicity
- Expedites the training process through efficient SVD computations.

### 3.2 Bias-Alleviating Low-Rank Adaptation (BA-LoRA)

**BA-LoRA: Method for Addressing Challenges in Large Language Models (LLMs)**
* **Catastrophic Inheritance**: challenges posed by biased large-scale training data in LLMs, leading to vulnerabilities, limitations, and adverse effects on downstream tasks
* **Regularizers for NLU Tasks:**
  * **Consistency Regularization (CR_NLU)**: based on mean squared error loss between normalized output logits of pre-trained and fine-tuned models to retain essential information during fine-tuning
  * **Diversity Regularization (DR_NLU)**: minimizes covariance matrix off-diagonal elements to encourage more diverse representational structures in LLMs and prevent encoding of similar samples
  * **Singular Value Decomposition Regularization (SVDR_NLU)**: enhances model generalizability by maximizing the sum of top k singular values to emphasize significant components
* **Overall Objective Function for NLU:**
  * Formulated as a combination of task loss, consistency regularization, diversity regularization, and SVD regularization
* **Regularizers for NLG Tasks:**
  * **Consistency Regularization (CR_NLG)**: uses Kullback-Leibler Divergence to ensure the fine-tuned model retains knowledge from pre-training
  * **Diversity Regularization (DR_NLG)**: increases entropy of predicted token distributions to encourage diverse outputs
  * **Singular Value Decomposition Regularization (SVDR_NLG)**: maximizes relative contribution of top k singular values to improve model focus on critical aspects of data
* **Overall Objective Function for NLG:**
  * Formulated as a combination of task loss, consistency regularization, diversity regularization, and SVD regularization.

## 4 Experiments

**Evaluation of Proposed BA-LoRA Method:**

- Assessed on NLG and NLU benchmarks
- Demonstrated superiority over existing LoRA variants
- Rigorous experimentation revealed effectiveness in mitigating noise data impact
- Enhanced model robustness and generalizability due to improved performance

### 4.1 Models and Datasets

**Evaluation Approach**
- Conduct experiments using various language models: LLaMA 2-7B, LLaMA 3-8B, Mistral-7B, Gemma-7B, GPT-2-XL (Generation)
- BERT-Large (BERT-L), RoBERTa-Large, DeBERTa-v3-base (Understanding)
- Ensures coverage across various architectures and parameter scales

**Datasets**
- Natural Language Generation: GSM8K, MATH, HumanEval, MBPP, MT-Bench
- Natural Language Understanding: GLUE benchmark, GLUE-X benchmark

**Rationale**
- Wide range of tasks in both NLG and NLU for thorough examination
- Assesses model performance and generalization capabilities.

### 4.2 Baselines

**Comparison of BA-LoRA**

- Compared to baselines: Full Fine-Tuning (Full FT) and LoRA [34], PiSSA [56]
- Differences in fine-tuning methods:
	+ Full FT: fine-tunes all model parameters, requiring the most resources
	+ LoRA [34]: inserts a low rank adapter $AB$ into linear layers
	+ PiSSA [56]: performs SVD on weight matrix $W$, initializing $A$ and $B$ based on singular value components

Note: Citations provided for each baseline method.

### 4.3 Implementation Details

**Experiment Strategies and Implementation Details**

**BA-LoRA Implementation**:
- Utilize PiSSA [^56] implementation strategy for loss computation
- Use Float32 computation type for both base model and adapter

**NLU Tasks Hyperparameters**:
- $\lambda_{1}=1e-4$ , $\lambda_{2}=4e-4$ , $\lambda_{3}=1e-4$
- lora\_r = lora\_alpha = 128
- AdamW optimizer with batch size 128, learning rate $2e-5$, cosine annealing schedules, and a warmup ratio of 0.03, without weight decay

**NLG Tasks Hyperparameters**:
- $\lambda_{1}=1e-4$ , $\lambda_{2}=3e-4$ , $\lambda_{3}=1e-4$
- lora\_r in {8, 16}, lora\_alpha = {8, 16}
- AdamW optimizer with linear learning rate schedule to optimize and tune LR from {$1e-4$, $2e-4$, $3e-4$, $4e-4$, $5e-4$, $6e-4$, $5e-5$, $3e-5$}
- Batch sizes (BS) selected from {6, 8, 16, 32}

**Appendix**: [B] Provides detailed hyperparameters used on GLUE benchmark

**Hardware**:
- Experiments conducted using NVIDIA A40 (48G) GPUs

### 4.4 Results and Analysis

**Performance Comparison of Various Models and Methods on NLG and NLU Tasks**

**NLG Performance:**
- BA-LoRA outperforms baseline methods on GSM8K, MATH, HumanEval, and MT-Bench tasks: 1.82%, 0.55%, 2.03%, 0.24% (respectively) improvement over PiSSA.
- Consistent performance advantages across NLG tasks indicate BA-LoRA's effectiveness in augmenting both generative and comprehension capabilities for language models.

**NLU Performance:**
- BA-LoRA outperforms baseline methods on GLUE benchmark: 0.59% to 1.61% improvement over PiSSA for RoBERTa-large and DeBERTa-v3-base, respectively.
- Consistent performance advantages across NLU tasks indicate BA-LoRA's effectiveness in mitigating catastrophic inheritance and improving model robustness.

**Analysis on Mitigating Noisy Data:**
- BA-LoRA outperforms LoRA on GLUE and GLUE-x benchmarks, reducing the negative impacts of noise in pre-training data: 2.03% to 2.47%, 1.61% to 1.90% improvement for BERT-L and GPT-2-XL respectively.

**Analysis on Mitigating Imbalanced Data:**
- BA-LoRA effectively alleviates the impact of imbalanced data in pre-training: clearer category separation in fine-tuned models using t-SNE visualization.

**Ablation Study:**
- Single regularization terms have varying degrees of performance improvement on GSM8K and MATH datasets: LLaMA-2-7B, Mistral-7B, Gemma-7B.
- BA-LoRA model with all regularization terms achieved the highest performance on both datasets, validating the effectiveness of the proposed regularization strategy.

## 5 Conclusion

**BA-LoRA: Bias-Alleviating Low-rank Adaptation**

- Novel parameter-efficient fine-tuning method for pre-trained language models
- Mitigates catastrophic inheritance
- Consists of three key components: consistency, diversity, and SVD regularization
- Preserves pre-training knowledge, enhances output diversity, and improves generalization
- Outperforms existing baselines on NLG and NLU tasks
- Robust to noisy and imbalanced data
- Ablation studies validate the effectiveness of each regularization term
- Potential for a general-purpose fine-tuning method
- Addresses key challenges in real-world deployment of pre-trained language models

## 6 Ethics Statement

* Study: Develop & evaluate BA-LoRA, a parameter-efficient fine-tuning method for LLMs, aiming to reduce bias and improve performance.
* Utilizes open-source datasets for both fine-tuning and evaluation across Natural Language Generation (MetaMathQA, CodeFeedback, WizardLM-Evol-Instruct) and Natural Language Understanding (GLUE, GLUE-X benchmarks).
* Committed to responsible AI development & application; monitoring ethical issues throughout research.

## 7 Reproducibility

**Reproducibility of Results:**

Detailed experimental setup described in Section [4.3](https://arxiv.org/html/2408.04556v2#S4.SS3 "4.3 Implementation Details ‣ 4 Experiments ‣ BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models") and Appendix Section [B](https://arxiv.org/html/2408.04556v2#A2 "Appendix B Details of Models and Datasets ‣ BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models"):
- Model & dataset introductions
- Hyperparameter configuration
- Evaluation procedures

All materials are publicly available. Improved implementation scripts and fine-tuning strategies for verification. Code and pre-trained model weights will be made public post-acceptance, ensuring transparency and reproducibility.

## Appendix A Background

### A.1 Challenges of Bias and Noise in Pre-training Data

**Machine Learning Challenges: Bias and Noise**
- Bias and noise in pre-training datasets pose a significant challenge to building dependable machine learning models
- Mislabeled data and imbalanced distributions can lead to underperformance on downstream tasks and reinforce existing biases [^69]
- This issue is exacerbated by large-scale datasets where manual curation is impractical, potentially introducing various inaccuracies and biases [^57]
- Real-world, instance-dependent label noise can cause models to learn these inaccuracies, resulting in poor generalization [^24]
- Addressing these challenges is crucial for advancing machine learning and ensuring models are both effective and equitable.

### A.2 Mitigating Bias and Noise through Parameter-Efficient Fine-Tuning Methods

**Pre-training Data Bias and Noise Mitigation:**

* Parameter-efficient fine-tuning techniques are promising solutions for adapting pre-trained models to new tasks while minimizing parameter updates, reducing overfitting risks.
* Methods include integrating lightweight adaptation modules, prefix tuning, low-rank adaptations, and selectively fine-tuning specific model components.
* These approaches promote efficient model refinement, preserve valuable pre-training representations, enhance performance on downstream tasks, improve generalization, reduce noise and bias, create more robust models, and contribute to fairer AI systems by addressing data quality issues directly.

## Appendix B Details of Models and Datasets

### B.1 Details of Models

**Pre-trained Language Models Used**

**Variety of Pre-trained Language Models:**
- Meta AI’s LLaMA-2-7B, LLaMA-2-13B, LLaMA-3-8B, and LLaMA-3-70B
- Mistral AI’s Mistral-7B-v0.1
- Google’s Gemma-7B
- Alibaba Cloud’s Qwen-1.5-7B
- 34B parameter Yi-1.5-34B
- DeepSeek-MoE-16B
- Mixtral-8x7B-v0.1

**Performance:**
- Good natural language generation tasks performance
- Medium-sized model efficiency (Mistral AI’s Mistral-7B-v0.1)
- Lightweight open-source model (Google’s Gemma-7B)
- Strong language understanding and generation capabilities
- High-level language tasks design (34B parameter Yi-1.5-34B)
- Increased capacity without significantly increasing computational costs (DeepSeek-MoE-16B)
- Efficiently utilizes active parameters to outperform larger models (Mixtral-8x7B-v0.1)

**Architectures and Pre-training Objectives:**
- BERT-Large and RoBERTa-Large: masked language modeling objective, BooksCorpus and English Wikipedia datasets
- GPT-2-XL: autoregressive language modeling objective, WebText dataset
- DeBERTa-v3-base: replaced token detection objective with Gradient Disentangled Embedding Sharing (GDES), diverse dataset comprising Wikipedia, BooksCorpus, OpenWebText, CC-News, and Stories.

### B.2 Details of Datasets

**Table 6: GLUE Benchmark Datasets and Evaluation Metrics**

**GLUE Benchmark**:
- Diverse set of natural language understanding tasks
- Includes grammatical acceptability (CoLA), sentiment analysis (SST-2), paraphrase detection (MRPC and QQP), sentence similarity (STS-B), natural language inference (MNLI, QNLI, RTE), and coreference resolution (WNLI)
- Number of training examples varies significantly across datasets
- Tasks involve binary or multi-class classification, with up to 5 classes in STS-B
- Evaluation metrics are tailored to each task and employ accuracy, F1 score, Matthews correlation coefficient, and Pearson/Spearman correlation coefficients where appropriate

**Table 7: Summary of GLUE-X Out-of-Domain Tasks for Transfer Performance Evaluation**

**GLUE-X Datasets**:
- Cover a broad spectrum of natural language understanding tasks
- Includes natural language inference (SNLI, HANs, SciTail, MNLI mismatched), sentiment analysis (IMDB), question answering (NewsQA), semantic relatedness (SICK), and grammatical error detection (Grammar Test)
- Each task involves binary classification
- Test sizes range from 9,832 samples (MNLI mismatched) to 570,152 samples (SNLI)
- Accuracy is the primary evaluation metric across most datasets, except for the Grammar Test which uses the Matthews correlation coefficient

### B.3 Specific Hyperparameter Settings of RoBERTa-large and DeBERTa-v3-base on GLUE

**RoBERTa-large and DeBERTa-v3-base Model Training**

**Hyperparameters**:
- **RoBERTa-large**:
  - MNLI: 10 epochs, batch size 32, learning rate $1\times 10^{-4}$ (LoRA\_alpha = 16)
  - SST-2: 10 epochs, batch size 32, learning rate $2\times 10^{-4}$ (LoRA\_alpha = 16)
  - Smaller datasets (MRPC, CoLA, RTE): 20 epochs, various batch sizes and learning rates, LoRA\_alpha = 8 or 16
- **DeBERTa-v3-base**:
  - MNLI: 5 epochs, batch size 16, learning rate $5\times 10^{-5}$ (LoRA\_alpha = 8)
  - SST-2 and MRPC: 20 epochs, various batch sizes and learning rates, LoRA\_alpha = 8
  - RTE: 50 epochs, batch size 16, learning rate $1\times 10^{-4}$ (LoRA\_alpha = 8)

**Dataset Specific Requirements**:
- Natural language inference
- Sentiment analysis
- Paraphrase detection
- Linguistic acceptability
- Semantic textual similarity

**Optimization**:
- Hyperparameters carefully selected to suit each dataset and task
- Ensuring rigorous and optimal training across various datasets.

### B.4 Specific Hyperparameter Settings of BERT-L and GPT-2-XL on GLUE and GLUE-x

**Training and Evaluation of BERT-Large (BERT-L) and GPT-2-XL Models on GLUE Benchmark Tasks:**
* Consistent performance ensured through training on GLUE benchmark tasks using three different random seeds per task for 10 epochs
* Hyperparameter search conducted over learning rates: {2×10^(-5), 3×10^(-5), 5×10^(-5)} with batch size of 32
* Adjusted training schedule based on dataset size: smaller datasets - 20 epochs, larger ones - 5 epochs
* Learning rates explored within {2×10^(-4), 3×10^(-4), 5×10^(-4)} for fine-tuning
* Parameters set with LoRA_rank = 8 and LoRA_alpha = 16, reduced batch size to 16 due to increased model complexity
* Adhered to Hugging Face Transformers guidelines for all other parameters

**Evaluation of GLUE-x Tasks:**
* BERT-L and GPT-2-XL models trained on GLUE were evaluated without further fine-tuning
* GLUE-x encompasses 13 out-of-distribution tasks, introducing domain shifts
* Models fine-tuned on SST-2 evaluated on IMDB, Yelp, Amazon, and Flipkart test sets for broader assessment of domain variability
* MNLI subset from GLUE used for t-SNE visualization due to diverse linguistic styles and label distributions
* Training limited to one epoch to expedite the process while still providing insights into differentiation between classes and sentence structures.

## Appendix C More Experiments

### C.1 Analysis on Different Sizes and Types of Models

**Experiment Comparison of Different Models on GSM8K and HumanEval Benchmarks**
- Figure 3: Performance comparison of various models on GSM8K and HumanEval benchmarks (source: arxiv.org)
- Includes LoRA, PiSSA, BA-LoRA, LLaMA-2-7/13B, LLaMA-3-8B/70B, Mistral-7B-v0.1, Gemma-7B, Qwen1.5-7B, Yi-1.5-34B, DeepSeek-MoE-16B, Mixtral-8x7B-v0.1
- BA-LoRA consistently surpasses LoRA and PiSSA across all models and tasks (Figure 4)

**Experiment Details:**
- Models fine-tuned on MetaMathQA-100K and CodeFeedback-100K datasets
- Evaluated on GSM8K and HumanEval benchmarks.

**Significance of BA-LoRA:**
- Superior ability to enhance model generalization compared to LoRA and PiSSA (Figure 3)
- Mitigates catastrophic inheritance in large language models.

### C.2 Evaluating the Performance of Different Ranks

**Performance Comparison of BA-LoRA, LoRA, and PiSSA**: The study evaluates these methods using LLaMA-2-7B and Mistral-7B-v0.1 models, fine-tuned for one epoch on MetaMathQA-100K dataset with ranks ranging from 1 to 128, tested on GSM8K and MATH datasets. Figure [4](https://arxiv.org/html/2408.04556v2#A3.F4 "Figure 4 ‣ C.1 Analysis on Different Sizes and Types of Models ‣ Appendix C More Experiments ‣ BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models") indicates that BA-LoRA outperforms LoRA and PiSSA across all rank settings and datasets. As the rank increases, both BA-LoRA and PiSSA surpass full parameter fine-tuning. Notably, BA-LoRA performs better, particularly on Mistral-7B-v0.1.

### C.3 t-SNE Visualizations of Feature Evolution during the Fine-tuning with LoRA and BA-LoRA

**Feature Evolution Comparison: LoRA vs. BA-LoRA (Fine-tuning)**

**Comparison of Class Separation during Fine-tuning:**
- **LoRA fine-tuning of BERT-L**: Slow class separation, scattered and overlapping distributions towards the end (Figure 5)
- **BA-LoRA fine-tuning of BERT-L**: Earlier and clearer separation, forming a distinct "Y" shape with defined boundaries (Figure 6)
- **LoRA fine-tuning of GPT-2 XL**: Scattered and overlapping clusters throughout training, minimal separation by final steps (Figure 7)
- **BA-LoRA fine-tuning of GPT-2 XL**: Clearer and distinct class separation, emerging earlier with pronounced boundaries (Figure 8)

**Visualizations:**
- Figure 5: t-SNE Visualization of LoRA Fine-Tuning of BERT-L [Link](https://arxiv.org/html/2408.04556v2/x8.png)
- Figure 6: t-SNE Visualization of BA-LoRA Fine-Tuning of BERT-L [Link](https://arxiv.org/html/2408.04556v2/x9.png)
- Figure 7: t-SNE Visualization of LoRA Fine-Tuning of GPT-2 XL [Link](https://arxiv.org/html/2408.04556v2/x10.png)
- Figure 8: t-SNE Visualization of BA-LoRA Fine-Tuning of GPT-2 XL [Link](https://arxiv.org/html/2408.04556v2/x11.png)

## Appendix D More Discussions

**Insights into Our Work**

Provided: Here, we offer further insights into our work. \n
Revised: Offer further insights on our work.

### D.1 Limitations

**BA-LoRA Improvements and Limitations**

* Focus on English language tasks: Limits generalizability to other languages/specialized domains
* Complexity in training due to computational overhead introduced by regularizers (consistency, diversity, SVD)
* Unknown impact on bias (fairness, societal stereotypes)
* Fixed selection and weighting of regularization terms across tasks may not be optimal for all scenarios

### D.2 Future works

**Future Research Directions for BA-LoRA:**

1. **Multilingual settings and specialized domains**: Assess applicability beyond English
2. **Optimization techniques**: Reduce computational overhead of regularizers to improve efficiency while maintaining performance gains
3. **Bias assessment**: Investigate impact on fairness, societal stereotypes, and other forms of bias
4. **Regularization terms**: Refine selection and weighting methods, possibly through automated or dynamic adjustment techniques
5. **Scalability**: Test on larger models with hundreds of billions of parameters and investigate integration with other bias mitigation strategies for synergistic effects
6. **Robustness improvement**: Further enhance model robustness through testing and research.

