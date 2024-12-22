# SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture

Jiayi Han\*, Liang Du, Hongwei Du, Xiangguo Zhou, Yiwen Wu, Weibo Zheng, and Donghong Han: authors of the paper "On De-identification of Medical Images"

\* indicates corresponding author.

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related work](#2-related-work)
  - [2.1 Parameter-efficient fine-tuning (PEFT)](#21-parameter-efficient-fine-tuning-peft)
  - [2.2 Task incremental learning](#22-task-incremental-learning)
- [3 Method](#3-method)
  - [3.1 Overview of LoRA-styled adapters](#31-overview-of-lora-styled-adapters)
  - [3.2 Mixture of experts as soft «math»R̂«/math»](#32-mixture-of-experts-as-soft-mathr̂math)
  - [3.3 Weight yielding with sliding clustering](#33-weight-yielding-with-sliding-clustering)
  - [3.4 Mixing LoRA adapters with dynamic merging](#34-mixing-lora-adapters-with-dynamic-merging)
  - [3.5 Summarising the proposed SLIM](#35-summarising-the-proposed-slim)
- [4 Experiments](#4-experiments)
  - [4.1 Implementation details](#41-implementation-details)
  - [4.2 Comparison with SOTA approaches](#42-comparison-with-sota-approaches)
  - [4.3 Ablation study](#43-ablation-study)
  - [4.4 Case study on the GSM8K dataset](#44-case-study-on-the-gsm8k-dataset)
- [5 Conclusion](#5-conclusion)
- [6 Limitations](#6-limitations)
  - [6.1 Performance degradation still exists](#61-performance-degradation-still-exists)
  - [6.2 Difficulty for implementation](#62-difficulty-for-implementation)

## Abstract

**Proposed Approach for Training LLMs:**
- Balancing training budget, downstream performance, and general capabilities is a challenge
- Training whole model for downstream tasks is expensive, can result in catastrophic forgetting
- Parameter-efficient fine-tuning (PEFT) reduces training cost but suffers from forgetting and limits learning on downstream tasks
- Proposed novel Mixture of Experts (MoE) framework: Soft LoRA and Identity Mixture (SLIM)
  - Dynamic routing between LoRA adapters and skipping connections
  - Suppression of forgetting through weight-yielding with sliding clustering
  - Convert mixture of low-rank adapters to model merging formulation
  - Fast dynamic merging of LoRA adapters to maintain base model's general capabilities

**Benefits:**
- Comparable to state-of-the-art PEFT approaches on downstream tasks
- Leading performance in mitigating catastrophic forgetting.

## 1 Introduction

**Large Language Models (LLMs)**
- Demonstrated extraordinary general capabilities
- Widely used for question-answering, code generation, mathematics, and other fields
- Increasing utilization in specialized domains
- Need to support tasks with a mixture of world and specific knowledge

**Challenges with LLMs**
- Expensive fine-tuning on large models
- May not be affordable for many users
- Catastrophic forgetting after fine-tuning on downstream tasks

**Approaches to Mitigate Catastrophic Forgetting**
1. **Low-rank adaptation (LoRA)**: A parameter-efficient fine-tuning (PEFT) approach that forgets fewer pieces of knowledge compared to full weight fine-tuning, with limitations in learning capacity [^15] [^3].
2. Optimizing model in orthogonal subspace [^33].
3. Inserting calibrating modules [^23].
4. Utilizing learnable mask prompts to fine-tune CLIP without loss of text-vision alignment [^3].
5. Data replay for continual training [^11] [^41].
6. Proposed **SLIM**: A novel MoE architecture with Soft LoRA and Identity Mixture.
   - Introduces identity layers as experts to route samples out of the downstream distribution to the identity layers, avoiding influence of downstream tasks.
   - Uses weight yielding with sliding clustering to estimate the distribution of samples from the downstream tasks and modify routing weight based on consistency between input sample and distribution.
   - Converts low-rank adaptation to model merging task and proposes a fast merging approach that dynamically merges mixture of LoRA adapters to the base model.
7. **SLIM**'s main contributions:
   - Proposes SLIM, a novel MoE PEFT algorithm that achieves comparable performance to SOTA PEFT methods while effectively alleviating catastrophic forgetting.
   - Proposes weight yielding with sliding clustering for dynamic routing between identity layers and LoRA adapters.
   - Proposes dynamic merging that converts MoE low-rank adaptation to model merging, eliminating catastrophic forgetting without data replay.

## 2 Related work

### 2.1 Parameter-efficient fine-tuning (PEFT)

**Parameter-Efficient Fine-Tuning (PEFT)**
* Strategies: LoRA-styled, adapter insertion, prefix tuning

**Prefix Tuning**
- Introduces learnable prompts and re-parameterization to transformer layers for adaptations [^22]
- P-tuning v2 removes re-parameterization [^25]
- Assigns learnable tokens to initial word embedding layer for training efficiency [^26], [^20]
- SMoP: short soft prompts for efficient training [^5]

**Adapter Modules**
- Introduce additional adaptive layers to pre-trained model [^14]
  - Serial Adapter: adapter modules added to self-attention and FFN layers [^14]
  - AdapterFusion: adapters inserted only after normalization layers for FFNs [^29]
- Parallel adapters introduced to transformer layers [^12]

**LoRA-styled Approaches**
- Assume change of full-weight matrix can be approximated by low-rank matrix [^15]
  - AdaLoRA: estimates parameters' importance and modifies adapters' rank [^40]
  - DoRA: decomposes pre-trained weights into scale and direction, updating only the direction [^24]
  - MoRA: increases rank of adapters with same number of parameters [^18]
  - LoRAMoE and MixLoRA: mixing LoRA adapters with dynamic routing [^9], [^21]
    * LoRAMoE: introduces contrastive loss function to mitigate knowledge forgetting during downstream training.

### 2.2 Task incremental learning

**Task Incremental Learning: Mitigating Knowledge Forgetting**

**Replay-Based Methods:**
- **Experience Replay (ER)**: Maintains a memory bank of previous tasks during training on new tasks to maintain stability. [^30]
  * Behavior cloning and off-policy learning from the replay for improved training.
  * Sampling from the memory bank for subsequent batches.
- **Generative Replay**: Similar to ER but focuses on generating new samples instead of reusing old ones.

**Regularization-Based Methods:**
- **Fine-tuning**: Fine-tune the base model on an orthogonal subspace to avoid influence from previous tasks. [^3]
  * Proposed by [^30]: Leverages behavior cloning and off-policy learning for training stability.
- **Contrastive Loss**: Alleviates loss of world knowledge during fine-tuning. [^9]
- **Adam-NSCL**: Forces network parameters to lie in the null space of previous tasks for capability reservation. [^37]

## 3 Method

**Figure 2: Overall Framework**

The proposed approach consists of an identity layer and LoRA adapters. The input series are routed to each identity layer based on their predicted weights and distances from the clusters. The weight is modified by the distance and yielded by the weight yielder, and the top-K experts are selected according to this weight. The final output replaces the weighted sum of MoE methods with dynamic merging.

### 3.1 Overview of LoRA-styled adapters

**LoRA Adapters for MLP Layers**

**Overview**:
- LoRA adapters modify output of MLP layers using low-rank learnable matrices
- Impact generalization of LLMs, but do not directly change base model weights
- Can be discarded when not needed to recover generalization

**LoRA Adapter Equation**:
- `y = Wx + b + LoRA(x)`
- `LoRA(x) = BAx`, where `r << min(d_1, d_2)` and `b` is bias

**Impact on Generalization**:
- LoRA adapters influence output of MLP layers
- Impacts generalization of LLMs

**Discarding LoRA Adapters**:
- Alternative implementation: `y = (W + R̂BA + (1 - R̂)0)x + b`
- `R̂: R^C → {0, 1}` is routing operation determining adapter discard

**Dynamic Approach**:
- Users cannot decide on individual instruction use of LoRA adapters
- Proposed dynamic approach for determining when to discard adapters

### 3.2 Mixture of experts as soft «math»R̂«/math»

**MoE Architecture for LoRA Adapters:**
- Introduces a routing layer to assign input tokens to multiple LoRA adapters (Mixtral models)
- Routing layer, R, assigns tokens to experts based on Softmax function
- Top K experts process the input, Eq. [3] and Eq. [4]:
  * ŵ_i = [Softmax(R(x))_i, if ℛ(R(x)_i) <= 2; 0, otherwise]
  * Select top-K LoRA experts for processing
- Learnable routing layer replaces expert routing operation R̂ with a conditioned one on the input, Eq. [5]:
  * y = Wx + b + ∑_iR̂(x)_if_i(x)
  * f_i: LoRA adapter or identity layer satisfying f(x) = x
  * R̂(x): routing weight satisfying ∑_iR̂(x)_i = 1
- Transformation of Eq. [6]:
  * By setting R̂(x)_i = ŵ_i/Z, it transforms to the same formulation as Eq. [4]

**Soft LoRA and Identity Mixture:**
- Method for SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture
- Multiple experts introduced in Eq. [5]:
  * y = Wx + b + ∑_iR̂(x)_if_i(x)
  * f_i: LoRA adapter or identity layer satisfying f(x) = x
  * R̂(x): routing weight satisfying ∑_iR̂(x)_i = 1
- Transformation of Eq. [6]:
  * By setting R̂(x)_i = ŵ_i/Z, it transforms to the same formulation as Eq. [4]

### 3.3 Weight yielding with sliding clustering

**Clustering Approach for Samples Occupied from Target Domain:**
* Mixture of Gaussian distribution assumed for input sample distribution
* Calculate distance of each sample to nearest cluster center
* Large distances indicate samples less likely from target domain, assigned to identity expert
* Cluster assignment and update during training:
  + Randomly initialize N cluster centers (c_i in R^C)
  + For each input sample x in R^N,C :
    - Assign nearest cluster based on distance to its center
    - Update center and variance of each channel
* No cluster updates during inference
* Routing weights calculation:
  + Rnorm(x)_i = routing logits for input x and expert i
  + d = max(|x-c_idx|/σ_idx) - 1, where idx is the index of nearest cluster
  + w_i = [Softmax(w_i), if w_i <= K; 0 otherwise]
* Where:
  + TYPE(·) returns layer type
  + Rnorm(x)_i = routing logits for input x and expert i calculated using the nearest cluster center and variance
  + K represents the number of experts to be activated.

### 3.4 Mixing LoRA adapters with dynamic merging

**DARE Findings and Proposed Approach**

**DARE Findings**:
- LLMs are robust to slight parameter changes
- Merging task vectors of fine-tuned LLMs can obtain multiple downstream capabilities

**Proposed Approach**:
- Convert mixing of multiple LoRA adapters to dynamic model merging
- Formulate θ_DARE as the weight of the merged model based on θ_PRE and task vectors
- Merge general instruction tuning model and downstream model trained by LoRA: θ_M = θ_PRE + λ/1-p(θ_Ins-θ_PRE) + λ/1-p(θ_Ins+BA-θ_PRE)
- Eliminate access to pre-training model during inference

**Fast Implementation**:
- Approximate random sampling without introducing extra computational cost
- Mask sub-matrices of the low-rank adapter: θ_M = θ_Ins + M_A⊗(BA)
- Denote row set of masked elements as 𝒮_row and column set as 𝒮_col, approximate masking as B'A' where MASK(a,b) masks rows of a according to index set b
- Randomly sample subset of 𝒮_row and 𝒮_col to fit original masking ratio
- Introduce L1 regularization to adapters: ℒ_L_1 = ∑l∑k(B_k^l/r x C_out + A_k^l/rx C_in)

### 3.5 Summarising the proposed SLIM

**SLIM: Mitigating Catastrophic Forgetting with Soft LoRA and Identity Mixture**

**Modifications to LoRA Architecture:**
- **Mixture of Experts (MoE) architecture**: Utilized as soft R̂ in Eq. [2]
  * Identity layers along with LoRA adapters as experts
- **Weight Yielding with Sliding Clustering**: Correct routing weight and encourage prediction of downstream tasks
- **Dynamic Merging Approach**: Convert MoE to model merging formulation to fuse base model's general capacity and adapter's downstream capability

**Architecture Overview:**
1. Input assigned to nearest cluster center
2. Router predicts routing weight based on distance
3. Yielding and normalizing routing weight (Eq. [9] & [10])
4. Activating mixed experts (identity layers, LoRA adapters) according to the routing weight
5. Obtaining final output through dynamic merging

**Equations:**
- y = Wx + b + ∑i w\_if\_i(x)
- f\_i(x) = [x, i <= K; B^'_iA^'_ix, otherwise]

## 4 Experiments

### 4.1 Implementation details

**Evaluation of SLIM on Openchat-8B Model**

The SLIM model was evaluated using the Openchat-8B model, based on Llama3-8B-Instruct [^36]. Two training settings were conducted: single dataset setting (SDS) and multi-dataset setting (MDS). In SDS, the model trained with one downstream task. In MDS, multiple datasets were mixed for training, including OBQA [^27], SIQA [^32], and BOOLQ [^6]. For SDS, CSQA [^34], Hellaswag [^39], Winogrande [^31], ARC-e, and ARC-c [^7] datasets were used. Fine-tuned models in MDS were assessed on general tasks like MMLU [^13], GSM8K [^8], and PIQA [^2]. The fine-tuning was done for 2 epochs, on a single NVIDIA-A100 80G GPU, with a batch size of 16.

### 4.2 Comparison with SOTA approaches

**Comparison of SLIM with Other PEFT Approaches**
* **LoRAMoE**, **MixLoRA**, **MoLA**: Set to rank=16, N_experts=8
* **SLIM**: Holds N_experts=10 with 2 identity layers as experts, set rank to 128 for fair comparison

**Performance on Downstream Tasks (Tab. 1)**
* SLIM achieves comparable performance on downstream tasks and best performance on multiple datasets
* Models fine-tuned under MDS setting evaluated on general datasets (Tab. 2)

**Generalization Capacity**
* Proposed SLIM method significantly alleviates loss of generalization capacity of the fine-tuned model without any extra data replay

**Comparison with SOTA Approaches on Downstream Tasks (Tab. 1)**
| Model       | CSQA     | HellaS    | WinoG     | ARC-c      | ARC-e      | OBQA       | SIQA       | SIQA      | BOOLQ      | AVG        |
|-------------|-----------|-----------|-----------|------------|------------|-----------|-----------|----------|----------|---------|
| **LoRA**     | 78.13     | 87.82     | 78.37     | 81.65      | 85.26      | 68.40      | 72.36      | 77.39    | 67.15     | 77.39     |
| **DoRA**     | 78.95     | 87.93     | 79.29     | 81.56      | 82.28      | 69.60      | 71.85     | 77.32    | 67.09     |            |
| **LoRAMoE**  | **88.78** | **94.61** | 84.21     | 84.21      | 89.68      | 87.00      | 77.53     | 74.98    | 85.12     |            |
| **MixLoRA**  | 85.01     | 93.68     | 85.08     | 82.33      | 85.48      | 84.00      | 77.89     | 73.10    | 83.06     |            |
| **MixLoRA-Dy**| 85.50     | 93.82     | 84.92     | 83.19      | 87.83      | 82.60      | 78.40     | 73.30    | 83.16     |            |
| **MoLA**     | 85.66     | 93.82     | 82.00     | 82.84      | 87.33      | 82.00      | 78.40     | 78.40    | 83.11     |            |
| **SLIM (Ours)** | **93.28** | **94.87** | 84.13     | **88.22**   | **91.83**   | **87.00**  | **81.57** | **86.63**| **81.52** | **87.33** |

**Comparison with SOTA Approaches on Generalization (Tab. 2)**
| Model       | MMLU     | GSM8K    | PIQA      | AVG        |
|-------------|-----------|----------|------------|---------|
| **LoRA**     | 32.73     | 0.00     | 60.99      | 31.24    |
| **DoRA**     | 31.07     | 0.00     | 57.18      | 29.41    |
| **LoRAMoE**  | 60.20     | 59.43    | 79.54      | 66.39    |
| **MixLoRA**  | 55.41     | 20.47    | 77.36      | 51.08    |
| **MixLoRA-Dy**| 56.14     | 21.38    | 78.43      | 51.98    |
| **MoLA**     | 53.15     | 15.54    | 74.70      | 47.79    |
| **SLIM (Ours)** | **65.83** | **76.11** | **84.65**  | **75.65** |

### 4.3 Ablation study

**Effects of Proposed Modules (SLIM)**
* Effectiveness:
  * LoRA adapters for attention layers removed: minor effect on MDS, reduces forgetting by 2.29%
  * Identity layers inserted: reduces forgetting by 0.96%, performance drop on MDS
  * Weight boosts model by 2.59% compared to baseline without LoRA
  * Dynamic merging significantly boosts the model on MMLU by 5.54% and on MDS
* Efficiency: fast implementation of dynamic merging reduces time cost while having a minor influence on performance (Tab. [3](https://arxiv.org/html/2410.07739v1#S4.T3))
* Masking rate: influences the contribution of LoRA experts, if too large results in significant performance drop; small masking ratio cannot inhibit catastrophic forgetting (Fig. [4](https://arxiv.org/html/2410.07739v1#S4.F4))
* General capacity degradation during fine-tuning: SLIM is more robust to the training process than LoRA and MixLoRA (Fig. [5](https://arxiv.org/html/2410.07739v1#S4.F5))
* Number of LoRA adapters in SLIM: a larger number of experts enhances downstream performance with no obvious influence on MMLU (Fig. [6](https://arxiv.org/html/2410.07739v1#S4.F6))

### 4.4 Case study on the GSM8K dataset

**DoRA, LoRAMoE, and SLIM Comparison:**
* **Question**: In a dance class of 20 students, percentage enrolled in hip-hop dance is asked
* **Temperature (t) and repetition penalty (p) settings**: t=0, p=0 for DoRA and LoRAMoE; p=1.5 for DoRA due to repeated output

**DoRA Results:**
- Converts GSM8K questions to multiple choice questions
- No answer provided for percentage of students enrolled in hip-hop dance

**LoRAMoE Results:**
- Generates formulated output but fails to understand the question about percentage
- No answer provided for percentage of students enrolled in hip-hop dance

**SLIM Results:**
- Capable of generating formulated output
- Provides an answer: 60% of entire students are enrolled in hip-hop dance

**DoRA vs LoRAMoE vs SLIM**:
- DoRA tends to convert GSM8K questions to multiple choice questions
- LoRAMoE and SLIM can generate formulated output but have different capabilities
- SLIM provides an answer for percentage of students enrolled in hip-hop dance.

## 5 Conclusion

**SLIM Proposal**

The SLIM method is a parameter-efficient fine-tuning approach that includes LoRA experts and identity layers as experts, while dynamically routing between LoRA adapters and identity layers. The routing weights are determined based on latent space distances. We also introduce dynamic merging to model merging, allowing for effective combination of downstream capabilities with the original pre-trained model capabilities. SLIM performs comparably to state-of-the-art PEFT methods while minimizing capability loss with fewer parameters.

## 6 Limitations

### 6.1 Performance degradation still exists

* The proposed method mitigates catastrophic forgetting but may still see a slight performance drop after fine-tuning. This could be due to random sampling being suboptimal for LLM merging and because the loss function is ineffective at addressing knowledge loss during downstream fine-tuning. Advanced exploration of sampling strategies and training objectives may offer solutions.

### 6.2 Difficulty for implementation

**Concise Version:**

* Various implementation and acceleration frameworks have been suggested for LLMs, such as vLLM [^35] and ollama [^28].
* Although LoRA adapters can be adjusted to the base model, both vLLM and LoRA support multiple LoRA adapters for a single base model, incorporating mixed heterogeneous adapters (LoRA adapters and identity layers) remains challenging.
* Our current focus is on developing an acceleration engine for MoE of LoRA adapters to facilitate its use.

