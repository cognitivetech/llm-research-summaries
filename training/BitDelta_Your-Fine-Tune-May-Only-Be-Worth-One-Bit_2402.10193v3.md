# BitDelta: Your Fine-Tune May Only Be Worth One Bit

"The paper available at `https://arxiv.org/html/2402.10193v3` discusses the concept of using Generative Models for Data Augmentation (GMDA). It introduces a novel method to enhance dataset size and improve data quality."

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
  - [2.1 Full Model Compression](#21-full-model-compression)
- [3 BitDelta](#3-bitdelta)
  - [3.1 Method](#31-method)
  - [3.2 Methodology Cost](#32-methodology-cost)
  - [3.3 Implication](#33-implication)
- [4 Experiments](#4-experiments)
  - [4.1 Setup](#41-setup)
  - [4.2 Accurate Quantization](#42-accurate-quantization)
  - [4.3 Latency Improvement](#43-latency-improvement)
- [5 Conclusion](#5-conclusion)
- [Appendix A Appendix](#appendix-a-appendix)
  - [A.1 Societal Impact](#a1-societal-impact)
  - [A.2 Additional Experiments](#a2-additional-experiments)

## Abstract

**Large Language Models (LLMs)**
- Typically trained in two phases: pre-training on large internet-scale datasets, fine-tuning for downstream tasks
- **Pre-training**: higher computational demand
- **Fine-tuning**: less new information, more compressible assumption explored

**Decomposing Weights of Fine-Tuned Models**
- Decompose weights into pre-trained components and an additional **delta**
- Introduce a simple post-fine-tuning method: BitDelta

**BitDelta Method**
- Successfully quantizes the delta down to 1 bit without compromising performance
- Highlights potential redundancy of information added during fine-tuning
- Significant implications for multi-tenant serving and storage of fine-tuned models

**Benefits of BitDelta**
- Reduces GPU memory requirements by more than **10x** in multi-tenant settings
- Decreases per-user generation latency by over **10x**

**Validation of BitDelta**
- Experiments across Llama-2, Mistral and MPT model families
- Shows minimal performance degradation on models up to 70B parameters.

## 1 Introduction

**Fine-Tuning Large Language Models (LLMs)**

**Pretrain-Finetune Paradigm**:
- Revolutionized machine learning
- LLMs effective for critical tasks like instruction following and alignment
- Performant on a wide array of niche, highly impactful applications

**Challenges in Fine-Tuning**:
1. **Expensive Storage**: Each new fine-tuned model is large, making it expensive to store and manage on disk.
2. **Expensive Serving**: Distinct fine-tuned models demand significant GPU memory, making it difficult and expensive to concurrently serve without noticeable downtime.

**Decomposing Fine-Tuning Weights**:
- Decompose the fine-tuned model weights into the base pretrained model and a **delta** induced by the fine-tuning process.
- Compress this delta while maintaining model performance to sidestep storage and GPU memory demands.

**Parameter-Efficient Fine-Tuning (PEFT)**:
- PEFT methods like LoRA enforce highly structured, compressed deltas during fine-tuning.
- Recent work shows PEFT methods may not match full parameter fine-tuning, especially on high resource tasks.

**BitDelta Approach**:
- Applies 1-bit quantization to the weight delta between fine-tuned and base models.
- Quantize the sign bits and a trainable high-precision scale factor for each weight matrix.
- Initialize the scale factors to achieve best approximation error and further refine with distillation steps.
- Reduces memory consumption in multi-tenant serving by representing multiple fine-tuned models with a single high-precision base model and multiple 1-bit deltas.

## 2 Related Work

### 2.1 Full Model Compression

**Quantization Techniques:**
* Used for reducing memory consumption and improving latency in large language models (LLMs) [^60]
* Rescaling between activations and parameters to mitigate outliers [^14]
* Decomposing matrix multiplications into 8-bit computations with additional 16-bit process for handling outliers [^19]
* Iteratively rounding weight columns to 3-4 bits of precision [^19][^33]
* Selective compression of crucial weights while compressing the majority [^29][^33]
* Focusing on a small significant set of weights for sparse, low-precision pattern [^4]
* Utilizing incoherence processing to quantize model weights to as low as 2 bits with minimal impact on performance [^5]

**Pruning:**
* Reduces memory consumption by pushing certain parameter values to zero, inducing sparsity in the model [^31]
* May fail to take advantage of modern hardware unless using structured sparsity patterns like 2:4 (50%) sparsity [^36]
* Demonstrated pruning method on LLMs that utilizes the 2:4 sparsity pattern and achieves a 50% sparsity ratio [^18]
* Challenging to obtain higher sparsity while being hardware-friendly.

**Early Work on Post-Training Delta Compression:**
* Some studies explore post-training delta compression using existing techniques like GPTQ, unstructured pruning [^22], or classic lossless compression algorithms [^26]
* Focuses on reducing the delta size to save storage [^26][^64]
* Utilizes pruning to improve model merging applications [^62]
* Reduces the size of PEFT modules to save storage [^47]
* Combines quantization with a low-rank approximation to reduce the delta size [^47]
* Concurrent and independent work explores using delta compression for multi-tenant serving, focusing on reducing model loading time from disk to GPU.

## 3 BitDelta

### 3.1 Method

**BitDelta Quantization Method**

**Composition:**
- Consists of two stages:
  1. **1-bit quantization**
  2. **Scale distillation**

**1-bit quantization:**
- **Weight delta**: $\Delta=W_{\text{fine}}-W_{\text{base}}$, representing modifications in weights post-finetuning
- Aim for efficient representation by encoding sign bits: $\hat{\Delta}=\alpha\odot\text{Sign}(\Delta)$
  - Sign function identifies positive or negative values of $\Delta$
  - Scaling factor $\alpha$: $1/nm\sum_{ij}|\Delta_{ij}|$
- Minimize quantization error in $L_{2}$ norm: $\left\|{\Delta}-\hat{\Delta}\right\|_{2}^{2}=\sum_{ij}{(|W_{ij}|-\alpha)}^{2}$
- Works on Llama-2 and Mistral families for various model sizes (7B to 70B parameters)
- Effective for SFT-based methods, RLHF-based methods, context extension methods (RoPE scaling)

**Scale distillation:**
- **Scaling factor** $\alpha$ plays a significant role in low-bit regime
- Per-matrix $L_{2}$ weight error is not perfect measure of overall model quality
- Optimize scales by performing model distillation: align output logits of quantized model to original fine-tuned model
  - Freeze model weights and optimize for objective: $\boldsymbol{\alpha}^{*}=\arg\min_{\boldsymbol{\alpha}}\mathbb{E}_{x\in X}{\left\|Z_{\text{fine}}(x)-Z_{\text{bin}}(x;\boldsymbol{\alpha})\right\|}^{2}$
- Robust to choice of calibration dataset $X$, as process is parameter efficient and crucial aspect is logit matching with fine-tuned model
- Distills on C4 dataset for 800 samples of length 128, using the same subset of C4 over all models to control for seed-based variations
- Uses Adam optimizer ($lr=10^{-4}$, $\beta=(0.9,0.999)$, $\epsilon=10^{-8}$) and 1x80 GB A100 GPU for 7B and 13B models, 6x80GB GPUs for 70B models
- Fast process: compresses 70B models in roughly 10 minutes.

### 3.2 Methodology Cost

**BitDelta vs Fine-tuning Methods**

**Advantages of BitDelta:**
- **Extremely cheap**: trains a single parameter per weight matrix
- **Efficient with shorter input sequences**: works on lengths as low as 128
- **Fewer training steps**: only requires 200 steps (batch size 4) compared to fine-tuning methods' 10,000-1,000,000 steps
- **Cost-effective**: similar to post-training quantization schemes like GPTQ and AWQ in terms of methodology cost.

**Distinction from Fine-tuning Methods:**
- **Training requirements**: BitDelta trains a single parameter per weight matrix versus thousands to millions for fine-tuning methods
- **Input sequence length**: BitDelta works on shorter sequences (128), unlike fine-tuning methods requiring longer context windows (4k, 8k)
- **Training steps**: BitDelta requires significantly fewer training steps (200) compared to fine-tuning methods.

### 3.3 Implication

**BitDelta: Improving Efficiency in Model Storage and Serving**

**Compressing Delta to 1-bit:**
- Enables multiple opportunities for improving efficiency
- Effective model storage: maintain base model + compressed deltas
- Facilitates model hot-swapping
  * Base model remains in GPU memory
  * Compressed deltas loaded dynamically based on requests
- Reductions in storage needs and loading times

**Multi-tenant Serving System:**
- BitDelta enables a multi-tenant serving system for general fine-tuned models
- Exploits GPU resource and saves inference cost when traffic is low or unbalanced

**Memory Consumption and Serving Latency:**
- Lower memory consumption leads to opportunity to improve serving latency
- Punica and S-LoRA examples: exploit LoRA's structure and memory saving
- Forward pass decomposition: $X^{\prime}_{i} = W_{\text{fine},i}X_{i}$
  * Base model weight and 1-bit delta computed separately
- Classical batched GEMM kernel for $W_{\text{base}}X_{i}$
- BitBLAS kernel for $\hat{\Delta}_{i}X$ in a batched setting, keeping deltas quantized until GPU cache transfer.

## 4 Experiments

### 4.1 Setup

**Comparison of Model Responses from Zephyr-7B- $\beta$ for Question 9 in MT-Bench, a Concise Advertisement Task**

**BitDelta-Initial vs. Baselines**:
- BitDelta-Initial is unable to follow instructions, producing an overly formal advertisement that exceeds the word limit
- With scale distillation, BitDelta successfully produces a concise, catchy advertisement slightly over the word limit

**Baselines**:
- Primary baselines are fine-tuned models without compression
- Comparison with 8-bit RTN, 4-bit GPTQ, and 2-bit QuIP on evaluations where BitDelta is run on quantized base models

**Models and Datasets**:
- Evaluated on eight tasks: MT-Bench, 25-shot ARC Challenge, 5-shot BBH, 10-shot HellaSwag, zero-shot TruthfulQA, zero-shot LAMBADA, zero-shot Winogrande, and 5-shot GSM8K
- Evaluated using FastChat for MT-Bench and lm-evaluation-harness for other tasks
- Denoted as "BitDelta-Initial" before scale distillation is applied

**Performance**:
- BitDelta performs well on aggregated metrics, even outperforming the baseline in many cases
- However, it's difficult to attribute performance to methodology or base model when both are performant
- Highlighted tasks like TruthfulQA, GSM8K, and MT-Bench where base models struggle to show that BitDelta accurately preserves fine-tune information

### 4.2 Accurate Quantization

**BitDelta Compression Technique**

**Compression Results**:
- Achieves over 10x compression of embedding and LM head layers
- Inconsistencies in tokenizer vocabularies prevent further compression

**Comparison with SVD Approximation**:
- BitDelta outperforms low rank approximation ($r=16$ and $r=128$) on Vicuna-7B v1.5
- Low rank approximation fails to fully capture fine-tune information, particularly in difficult multi-turn datasets like MT-Bench

**Main Results**:
- BitDelta is performant across various model families, sizes, and fine-tuning techniques
- Recovers all types of finetune information (SFT-based, RLHF-based, context extension)
- Scale distillation effectively raises scores on TruthfulQA, GSM8K, and MT-Bench

**BitDelta with Quantized Base Models**:
- BitDelta is performant when applied to quantized base models (8-bit RTN, GPTQ, QuIP#)

**Ablation over Fidelity of $\Delta$**:
- Successively applying BitDelta allows varying granularity of delta and assigning arbitrary scale factors
- TruthfulQA scores approach that of Vicuna-7B v1.5 as the fidelity of $\Delta$ increases

### 4.3 Latency Improvement

**BitDelta Model Latency Analysis**

**Background:**
- Analyzing decoding latency of BitDelta model
- Comparison with shared base weight backbone (Wbase) and S-LoRA
- Focus on batch size impact on latency

**Kernel Latency:**
- Benchmarking for typical low to medium batch sizes: $B\times N \ll N\times M$
- Backbone memory footprint effectively independent of batch size, unlike deltas
- BitDelta underperforms slightly in large-batch settings due to LoRA optimization

**Memory Usage:**
- Figure 4: Decoding latency comparison (left)
- Memory usage analysis for Llama 2-7B variants (right)
- Naive method maintains separate models per client leading to memory issues at higher batch sizes

**End-to-end Latency:**
- Comparison of BitDelta, S-LoRA and naive method
- BitDelta and S-LoRA share a single backbone, reducing memory usage and improving performance in larger batches
- Naive approach scales poorly with increasing batch size due to separate model computations

**Conclusion:**
- BitDelta and S-LoRA offer better memory utilization and lower decoding latency compared to naive method at larger batch sizes.

## 5 Conclusion

**BitDelta: A Simple, Efficient Approach to Quantifying Weight Delta during LLM Fine-tuning**

BitDelta represents weight delta using sign bits and a per-weight matrix scaling factor calibrated through distillation. This approach allows multiple fine-tuned models to be represented with one base model and 1-bit deltas, facilitating applications in multi-tenancy serving by reducing GPU memory requirements and improving generation latency. BitDelta is fast, accurate, and minimizes performance degradation, offering new avenues for efficient model deployment and resource utilization in machine learning.

## Appendix A Appendix

### A.1 Societal Impact

**Democratization of Fine-tuned Models**
- Reducing hardware requirements for serving fine-tuned models allows smaller entities to deploy state-of-the-art models more feasibly
- Accelerates innovation and application development across various industries and academic fields, making fine-tuned models accessible to a wider audience

**Dealignment Mitigation**
- BitDelta is a lossy compression method for fine-tune information in LLMs that may cause crucial alignment loss
- As BitDelta democratizes multi-tenant applications, dealignment concern may be exacerbated
- Encourage further work on evaluation techniques to detect alignment loss and create robust methods for its mitigation

### A.2 Additional Experiments

**Study Findings:**
* LoRA finetuned $r=16$ on UltraChat with minimal performance degradation using BitDelta (Table 7) [^17]
* BitDelta effective for parameter-efficient fine-tunes, not only full-parameter fine-tunes

**Table 8:**
* Results of applying BitDelta to quantized base models
* Detailed data in "arxiv.org/html/2402.10193v3#S4.T6"

**Table 9:**
* Ablation study on the fidelity of $\Delta$ (delta)
* Additional details in "arxiv.org/html/2402.10193v3#S4.F3"

**Table 10:**
* Results of BitDelta applied to fine-tuned models in Llama-2 and Mistral families
* Previous findings from Table 2 ["arxiv.org/html/2402.10193v3#S3.T2"]:
	+ 1-bit quantization method
	+ Detailed data in "arxiv.org/html/2402.10193v3#S3.T2"

