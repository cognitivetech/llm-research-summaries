# LoRA vs Full Fine-tuning: An Illusion of Equivalence

**Authors**: Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma (MIT CSAIL)

**Source**: [arXiv:2410.21228v1](https://arxiv.org/html/2410.21228v1)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background \& Related Work](#2-background--related-work)
  - [Methods for fine-tuning.](#methods-for-fine-tuning)
  - [LoRA, formally.](#lora-formally)
  - [LoRA Variants.](#lora-variants)
  - [Analysis of Solutions.](#analysis-of-solutions)
  - [Singular Value Decomposition.](#singular-value-decomposition)
- [3 Model Differences Between LoRA and Full Fine-Tuning](#3-model-differences-between-lora-and-full-fine-tuning)
  - [Intruder Dimensions Persist in Large-Scale Language Models](#intruder-dimensions-persist-in-large-scale-language-models)
- [4 Behavioral Differences Between LoRA and Full Fine-Tuning](#4-behavioral-differences-between-lora-and-full-fine-tuning)
- [5 Why Do Intruder Dimensions Exist?](#5-why-do-intruder-dimensions-exist)
- [6 Conclusion](#6-conclusion)

## Abstract

**Fine-Tuning Paradigms for Pretrained Models: An Analysis**

**Background:**
- Importance of fine-tuning pretrained large language models for downstream tasks
- Recent methods like Low-Rank Adaptation (LoRA) match full fine-tuned model performance with reduced trainable parameters

**Question:**
- Are solutions from different fine-tuning methods equivalent?

**Approach:**
- Analyze weight matrices' spectral properties in pretrained models

**Findings:**
1. **Differences in Weight Matrices**:
   - Full fine-tuning and LoRA yield distinct weight matrices
   - Singular Value Decomposition (SVD) structures are dissimilar
2. **Generalization Behaviors**:
   - Fine-tuned models exhibit varied performance outside the adaptation task's distribution
3. **Intruder Dimensions in LoRA:**
   - New, high-ranking singular vectors appear during LoRA fine-tuning
   - Not present during full fine-tuning
4. **Impact on Pretraining Distribution**:
   - LoRA models with intruder dimensions become worse models of pre-training distribution
   - Adapt less robustly to multiple tasks sequentially
5. **Rank-Stabilized LoRA:**
   - Closely mirrors full fine-tuning when performing similarly on target tasks
6. **Conclusion**:
   - Models updated with LoRA and full fine-tuning access different parts of parameter space, even when they perform equally on the fine-tuned distribution.
7. **Why Intruder Dimensions Appear in LoRA:**
   - Further examination required to understand the reasons and minimize their effects.

## 1 Introduction

**LoRA vs Full Fine-Tuning: An Illusion of Equivalence**

**Introduction**:
- Adapting large pre-trained models to downstream tasks via fine-tuning is a computation- and data-efficient way to create domain-specific models
- However, full fine-tuning becomes increasingly challenging and expensive as pre-trained models grow larger
- Recently, parameter-efficient fine-tuning (PEFT) methods, especially low-rank adaptation (LoRA), have been shown to enable fine-tuning with only a fraction of the trainable parameters
- But it is unclear if the solutions learned by full fine-tuning and LoRA are equivalent

**Differences in Spectral Properties**:
- **Figure 1**: Spectral dissimilarities between full fine-tuning and LoRA. Full fine-tuning retains most of the pre-training structure, while LoRA introduces intruder dimensions.
- **Figure 2a**: Measuring changes to the singular value decomposition (SVD) of pre-trained weights during fine-tuning. LoRA introduces intruder dimensions not present in full fine-tuning.
- **Figure 2b**: Comparing a matrix fine-tuned with full fine-tuning or LoRA shows differences in the singular vectors.
- **Figure 2c**: Comparing normal singular vs. intruder dimensions to all pre-trained singular vectors highlights the presence of intruder dimensions in LoRA.

**Differences in Generalization Behavior**:
1. **Structural differences**: LoRA and full fine-tuning produce different parameter updates, characterized by the existence of intruder dimensions. Full fine-tuned models remain spectrally similar to the pre-trained model and do not contain intruder dimensions.
2. **Behavioral differences**: LoRA fine-tuned models with intruder dimensions forget more of the pre-training distribution and exhibit less robust continual learning compared to full fine-tuning, despite matching accuracy in the adaptation task's distribution. However, higher-rank LoRA models (r≤8) perform better on these measures when using a high-rank (r=64) parameterization and rank-stabilizing the updated models.
3. **Rank stabilization**: Even with a high-rank parameterization, the LoRA updated models must be rank-stabilized to take advantage of higher ranks.

## 2 Background & Related Work

### Methods for fine-tuning.

**Pre-trained Language Models and Fine-Tuning**

* Pre-trained models provide a base for various applications, avoiding re-training from scratch [^19].
* Full fine-tuning (updating every model parameter) has been applied for adaptation [^4].
* Low Rank Adaptation (LoRA; [^9]) reduces computational and memory requirements by representing weight updates as a product of low-rank matrices.
* Previous research indicates that LoRA matches full fine-tuning performance on tasks like sequence classification [^9], instruction tuning [^3], and chatting [^3]. However, some studies have revealed underperformance for more challenging tasks such as code generation [^2].
* This note applies the observations of structural differences in cases where LoRA does not adapt well to an adaptation task compared to full fine-tuning, even when models achieve similar accuracy.

### LoRA, formally.

**LoRA vs Full Fine-Tuning: Differences and Tradeoffs**

**Pre-trained Weight Matrix W\_0**:
- m x n dimensions

**Full Fine-Tuning**:
- Treats learned matrix update as ΔWinR^mxn
- Equivalent to training all parameters in W\_0
- Requires large amount of computational resources

**LoRA (Learning with Relevance Annotation)**:
- Decomposes ΔW into a product of two matrices: BA
- BinR^mxr, AinR^rxn
- r is generally r≪min(m,n)
- During prediction: Y = (W\_0 + α/rBA)X

**Initialization**:
- B initialized to zero
- A sampled from an isotropic Gaussian
- All parameters in B and A are trained

**Trainable Parameters**:
- Full fine-tuning: mn trainable parameters per weight matrix
- LoRA: mr + rn trainable parameters

**Derivation of Gradients**:
- Refer to Appendix D for derivation

**Singular Vectors**:
- In fine-tuned models with LoRA: shift in singular vectors due to intruders dimensions (blank columns)
- No such shift found in full fine-tuning

### LoRA Variants.

**LoRA Variations**
- Numerous LoRA versions exist
- Improvements focus on initializing with principal components of weight matrix [^18], quantized training [^3], adaptive rank allocation [^36], or sequential training [^34]. Some propose alternative architectures [^16].
- This work focuses on the original LoRA setup [^9] while leaving a comprehensive analysis of variations and their effects for future research.
- Setting α=2r has empirically shown to enhance results for higher ranks [^2, ^11], which is our choice for most experiments in this paper.

### Analysis of Solutions.

**LoRA (Low-Rank Adaptation) vs Full Fine-Tuning:**

**Introduced by [^15]**:
- Intrinsic dimension measure used to argue that LoRA has low intrinsic rank
- Explained why only a small number of trainable parameters are necessary for 90% of full fine-tuning performance

**[^9]'s Hypothesis**:
- LoRA works because solutions of low intrinsic rank exist

**Comparison between LoRA and Full Fine-Tuning**:
- No past work has compared the rank or properties of weight matrices for both methods on matched tasks
- **[^16]**: LoRA has difficulty changing directional and magnitude components of a neuron independently, while full fine-tuning does not
    - Unclear if this difference is due to inability of LoRA to fit the adaptation task as well

**LoRA vs Full Fine-Tuning: Recent Findings**:
- **[^2]**: LoRA forgets less on previously learned information and more closely resembles pre-trained model
    - However, experiments in current study show opposite trends
- Significant differences in datasets used for evaluation between studies
- **[^5]**: LoRA more closely resembles pre-trained model

**Current Study**:
- Comparison of LoRA and full fine-tuning on sequence labeling tasks where they achieve the same performance
- Focuses on generalization behavior at a fixed target task accuracy.

### Singular Value Decomposition.

**Singular Value Decomposition (SVD)**
- Breaks down matrix M \_(m x n)\_ into three components: U \_(m x m)\_, V\_(n x n)\_ (orthonormal columns), and Σ \_(m x n)\_ (diagonal matrix)
- Matrix M can be represented as M = UΣV^T, where U and V^T represent rotations performed by M, and Σ represents scaling along those axes.
- Singular values in Σ, ranked in order, capture the most important axes of transformation that the matrix performs.

## 3 Model Differences Between LoRA and Full Fine-Tuning

**Singular Value Decomposition (SVD) and Fine-tuning**:
* Inspired by [^24]'s findings that SVD can be used to prune singular vectors for improved model performance
* This paper adopts the SVD of neural network parameters as a lens to understand changes made during fine-tuning

**LoRA vs Full Fine-tuning**:
* Measured by their cosine similarity with pre-trained singular vectors
* Visually observed differences in Fig. 1 and Fig. 3:
    * LoRA and full fine-tuning have different similarities to pre-trained singular vectors
    - LoRA appears to introduce "new" dimensions (intruder dimensions) with low cosine similarity to any pre-trained singular vector

**Intruder Dimensions**:
* Defined as a singular vector from the fine-tuned weight matrix with low cosine similarity to all pre-trained singular vectors
* Examples can be seen in Fig. 3, where LoRA appears to have "blank" columns (intruder dimensions) that are not present in full fine-tuning

**Model Differences**:
* LoRA introduces new singular vectors with large contributions to the norm of updated parameter matrices
* Full fine-tuning makes small changes to existing singular vectors and values

**Experiment Setup**:
* RoBERTa-base, a pre-trained encoder-only language model, fine-tuned on six different sequence classification tasks
* Computed the total number of intruder dimensions in each model

**Determining Intruder Dimensions**:
* For each top k highest-ranking singular vectors, measure maximum cosine similarity with all pre-trained singular vectors
* If less than some threshold ϵ , classify as an intruder dimension
* Repeat for different values of k and ϵ to verify robustness of findings

**Results**:
* LoRA models consistently contain intruder dimensions when rank r≤16, particularly for low values of ϵ
* Fully fine-tuned models rarely contain intruder dimensions in their top 10 singular vectors for ε values of about 0.6 to 0.9 across different settings
* The number of intruder dimensions drops as rank increases, suggesting they are induced by the low-rank nature and update rule of LoRA.

### Intruder Dimensions Persist in Large-Scale Language Models

**LoRA vs Full Fine-Tuning: An Illusion of Equivalence**

**Differences between LoRA and Full Fine-Tuning**:
- **Intruder dimensions**:
  - Present even in tasks where LoRA models learn less than full fine-tuning
  - Increase with higher rank (r=256) for LLaMA-7B and LLaMA2-7B models
  - Disappear as rank increases past a threshold, and LoRA begins to resemble full fine-tuning
  - More present in top 10 singular vectors of full fine-tuned Magicoder model compared to LoRA

**Effective Rank**:
- Effective rank of full fine-tuning solutions has significantly higher effective rank than LoRA updates, even with high adapter ranks and rank stabilization
- Across layers, effective rank of LoRA updates is less than half that of full fine-tuning and a quarter of the adapter rank
- Suggests LoRA is underutilizing its full capacity (r)

**Distribution of Intruder Dimensions**:
- Present throughout weight matrix, not just high or low singular values
- Number of intruder dimensions remains consistent regardless of fraction of fine-tuned singular vectors examined
- Full fine-tuning may change lower-ranking singular vectors more than LoRA in some cases

**Changes in Intruder Dimensions during Training**:
- Intruder dimensions gradually increase their "rank" (singular value) and change direction as training progresses
- Scaling α with the rank of the LoRA update reduces the number of intruder dimensions and increases effective ranks of matrices.

## 4 Behavioral Differences Between LoRA and Full Fine-Tuning

**Behavioral Differences Between LoRA and Full Fine-Tuning**

**Experiments:**
- Continual Learning performance evaluated for LoRA and full fine-tuning on multiple tasks (MNLI, QQP, SST-2, SIQA, Winogrande, FEVER) in specific dataset order
- After training on a task, LoRA weights are merged into the model and reinitialized before training on next task to examine performance without impacting model itself.

**Results:**
- Initial performance of LoRA matches full fine-tuning, but smaller ranks show greater degradation during continual learning
- Low ranks of LoRA forget previously learned tasks more and adapt less robustly in continual learning environment
- U-shaped curve identified for optimal rank for fitting to downstream task distribution while forgetting pre-training distribution least: r=64 minimizes forgetting of pre-training distribution.

**Findings:**
- LoRA models with very low or high ranks exhibit greater forgetting on pre-training distribution relative to full fine-tuning
- Models fine-tuned with LoRA when r=1 suffer from intruder dimensions and more forgetting than r=64 which had no intruder dimensions.
- Forgetting behavior decreases as rank increases, ranks r=8 and r=64 forget less than full fine-tuning, while ranks on extremes forget more.

**Settings α:**
- Continual learning and pre-training forgetting experiments repeated with fixed α instead of rank-scaled α
- LoRA models forget much more of both pre-training distribution (MNLI, QQP, FEVER) and previously learned tasks during continuals learning when α=8 instead of α=2r.

## 5 Why Do Intruder Dimensions Exist?

**LoRA (Large-batch Offline Rank Adjustment) vs Full Fine-tuning: Differences and Impact on Intruder Dimensions**

**Introduction to LoRA:**
- Introducing random vectors to a pre-trained matrix creates an intruder dimension
- Singular Value Decomposition (SVD) comparison between W + λvv^T and W
- Expectation of creating intruder dimensions when updating with LoRA

**Mathematical Conditions:**
- Comparison of SVD(W+λvv^T) and SVD(W)
- Pre-trained weights in R^nxn (W), randomly sampled vector in R^n (v), scalar value greater than largest singular value of W (λ)
- Nearly orthogonal to existing singular vectors

**Update Rules:**
- LoRA and full fine-tuning have different update rules
- LoRA uses larger learning rate, gradients projected into low-rank space
- Conditions similar to toy example above

**Impact of Freezing A:**
- Figure 10: Impact on number of intruder dimensions when only training B
- Randomly initialize A with singular values 1 and freeze it
- Compare with vanilla LoRA (train A and B separately)
- Sharp drop in high ranking intruder dimensions

**Product Parameterization:**
- Multiplication amplifies spectral differences between matrices
- Impact on introduction of intruder dimensions: Figure 10
- Matrix product of LoRA an important component due to spectral amplification effect.

## 6 Conclusion

**LoRA and Full Fine-Tuning Performance Differences**

* LoRA and full fine-tuning can have disparate generalization behaviors outside their respective fine-tuning task distribution.
* These methods create models with distinct spectral properties in weight matrices: LoRA models often feature "intruder dimensions," high-ranking singular vectors approximately orthogonal to the singular vectors of pre-trained weight matrices.
* The presence of intruder dimensions correlates with fine-tuned models forgetting more of the pre-training distribution and more during sequential training in a continual learning setup.

