# Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning

https://arxiv.org/html/2401.04151

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Our Method](#3-our-method)
  - [3.1 Preliminaries](#31-preliminaries)
  - [3.2 Chain of LoRA](#32-chain-of-lora)
- [4 Convergence of COLA and the Nonconvex Frank-Wolfe method](#4-convergence-of-cola-and-the-nonconvex-frank-wolfe-method)
- [5 Experimental Setup](#5-experimental-setup)
  - [5.1 Models and Tasks](#51-models-and-tasks)
  - [5.2 implementation details](#52-implementation-details)
- [6 Results and analysis](#6-results-and-analysis)
  - [6.1 Main Results](#61-main-results)
  - [6.2 Ablation Study](#62-ablation-study)
- [7 Conclusions and future work](#7-conclusions-and-future-work)

## Abstract

**Fine-tuning Methodologies for Pre-trained Large Language Models:**

**Low-rank Adaptation (LoRA)**:
- Encodes weight updates as product of two low-rank matrices
- Advantages: important for parameter-efficient fine-tuning with large models and diverse tasks
- Limitations: falls short of full-parameter fine-tuning for certain tasks in terms of generalization error

**Chain of LoRA (COLA)**:
- Iterative optimization framework inspired by Frank-Wolfe algorithm
- Bridges gap between LoRA and full parameter fine-tuning
- No additional computational costs or memory overheads
- Merges learned LoRA modules into pre-trained language model parameters
- Re-initializes optimization for new born LoRA modules

**Theoretical Convergence Guarantees and Empirical Results**:
- Provided to validate effectiveness of COLA algorithm

**Demonstration**:
- Across various models (OPT and llama-2) and seven benchmarking tasks
- Consistently outperforms LoRA without additional computational or memory costs.

## 1 Introduction

**Pre-trained Language Models:**
* Instrumental in natural language processing
* Demonstrate remarkable performance across various fields
* Large language model fine-tuning: adapting pre-trained models to specific tasks
	+ Improved performance on real-world applications (machine translation, code analysis)
	+ Challenges: computational expenses and memory requirements
* Parameter Efficient Fine Tuning (PEFT):
	+ Involves fewer adjustments to original model parameters
	+ Significantly reduces computational burden and time required for fine-tuning
* Low-Rank Adaptation (LoRA):
	+ Modifies a small, low-rank portion of the model's weights
	+ Advantages: significant reduction in computational costs
	+ Inferior to full parameter fine-tuning in terms of generalization error

**Investigating LoRA and Full Parameter Fine Tuning:**
* Goal: reduce generalization error gap between LORA and full parameter fine-tuning
* Method: learning a higher rank augmentation using residual learning (Chain of LORA, COLA)
	+ Inspired by Frank-Wolfe algorithm for matrix completion
	+ Iterative procedure to learn a low-rank addition to existing approximation
	+ Produces accurate higher rank completion
* Results:
	+ Fine-tuning OPT-1.3B with COLA brings a relative 6.47% test accuracy gain to LoRA on WSC
	+ LLama2-7B experiments show up to 4.4% relative test score improvement
* Theoretical analysis: demonstrates convergence to stationary points in smooth nonconvex optimization.

## 2 Related Work

**Parameter Efficient Finetuning Methods**

**Challenges**:
- Conventional full parameter fine-tuning becomes computationally impractical as model size and number of downstream tasks increase.

**Solutions**:
- **Adapter based methods**: Modify only a small portion of parameters while keeping the majority of pre-trained model parameters unchanged.
  * Adapter insertion between transformer layers
  * Only newly introduced adapters are trained during fine-tuning
  * Significantly enhances practicality and efficiency for diverse tasks
- **Prefix tuning methods**: Incorporate tunable parameters into input and hidden layers.
  * Offers reduction in memory load required for task-specific models
  * Outperforms full fine-tuning, particularly with limited data availability

**Adapter Based Approaches**:
- Bottleneck adapter module proposed by [^10] and placed twice within each transformer layer
  * Employs a bottleneck architecture with skip connection to constrain parameters
- Variant adapter architectures and placements proposed in concurrent work [^2]
- AdapterFusion: Two-stage learning framework for task-specific adapters [^20]

**Prefix Tuning**:
- Lightweight task-specific vectors, or "prefix," offered as reduction in memory load
- Efficient prompt tuning simplifies prefix tuning by concatenating a trainable tensor with model input embeddings [^14]

**LoRA and Variants**:
- LoRA: Introduces trainable low-rank matrices for weight update approximation during fine-tuning
  * Elaborated in preliminaries section below
- QLoRA: Leverages 4-bit quantization for efficient and effective fine-tuning of large language models [^4]
- Tied-LoRA: Incorporates weight tying and selective training for parameter efficiency [^23]
- LongLoRA: Extends context sizes of large language models with limited computation cost [^3]
- MultiLoRA: Designed specifically for better multi-task adaptation [^29]
- S-LoRA: Enhances scalable serving of multiple LoRA adapters [^24]
- Zero-order optimization methods proposed for fine-tuning large language models due to memory constraints [^19]

## 3 Our Method

**Fine-Tuning Method Description:**

1. The method consists of two parts: Background information and details of COLA (Figure 1).
2. Figure 1 displays an illustration of Chain of LoRA. Our approach begins by freezing a Language Model (LLM) and learning a sequence of low-rank matrices to simulate high-rank augmentation for task adaptation.
3. As shown in the dashed line box, the residual learning procedure is composed of three steps:
   - **Step 1**: Low-rank LoRA modules are fine-tuned.
   - **Step 2**: The learned LoRA weights are incorporated into the frozen model (Tie a knot).
   - **Step 3**: A new LoRA module is initiated, and the optimizer state is reset.
4. Steps 1-3 are repeated in this residual learning paradigm.

### 3.1 Preliminaries

**Low Rank Adaptation (LoRA)**
- Aims to improve efficiency of fine-tuning large language models by training smaller low-rank decomposition matrices for certain weights
- Hypothesizes "low intrinsic rank" of weight updates at task adaptation
- Injects trainable low-rank decomposition matrices into each layer of the Transformer architecture

**Weight Matrix Decomposition:**
- Pre-trained model weight matrix: $W_{frozen}$
- Weight update for task adaptation: $\Delta W$
- Low-rank decomposition: $BA$
  - Factor A: $\mathbb{R}^{r\times k}$
  - Factor B: $\mathbb{R}^{d\times r}$ (initially zero)
  - Rank r much smaller than min(d,k)

**Forward Pass:**
- $W_{frozen}x + \Delta Wx = W_{frozen}x + BAx$
- During training:
  * Freeze $W_{frozen}$ and optimize only $B$, $A$
- At deployment:
  * Merge learned low-rank matrices with frozen pre-trained model weights

**Frank-Wolfe Method:**
- Optimization algorithm for solving constrained convex or nonconvex optimization problems
- Handles constraints by using a linear optimization oracle instead of projections
- Iteratively finds a linear approximation of the objective function within feasible region and moves towards minimizer
- Suited for problems where linear optimization is easier than Euclidean projections
- Recently used in nonconvex optimization [^13]

### 3.2 Chain of LoRA

**Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning**

**Idea**:
- Form a chain of Low-Rank Adaptation (LoRA) modules and iteratively learn them

**Method Components**:
1. **Tune LoRA**:
   - Perform standard LoRA tuning
   - Learn only the A and B matrices, leaving other parameters untouched
2. **Tie a Knot**:
   - Merge the trained LoRAs into the frozen pre-trained model weights
3. **Extend the Chain**:
   - Re-initialize new LoRA modules to learn the residual weights update

**Algorithm**:
- Input: frozen pre-trained weights, chain knots, finetuning dataset, training objective, total iterations
- Initialize LoRA params and run for the specified number of iterations
  - Sample a minibatch for each iteration where LoRA is tied
    - Tie knot: Merge LoRAs to backbone weights
    - Extend chain: Re-initialize LoRA params
  - Perform forward pass, backward pass, and update LoRA params

**Approximation**:
- Approximate the optimal weight update $\Delta W^{\star}$ with a chain of low-rank matrix decompositions $(A_{i},B_{i})$

**Challenges**:
- Fine-tuning each $(A_i, B_i)$ is an easier optimization problem compared to learning $\Delta W^\star$ from scratch
- Hypothesis: Sum of low-rank tuples approximates $\Delta W^\star$ better than a single LoRA update $BA$

**Benefits**:
- COLA forms a chain of LoRAs, which may approximate the optimal residual weight updates more effectively
- Less computation compared to the baseline LoRA

**Training and Inference Cost**:
- Training cost determined by the rank of LoRA modules used in the chain
- Same training computation as LoRA when the rank is the same
- Lowering the rank of LoRAs may reduce overall training cost
- No latency overhead during inference since all learned $B_jA_j$ can be integrated into the original model weights.

## 4 Convergence of COLA and the Nonconvex Frank-Wolfe method

**The COLA Algorithm and Its Relationship to the Frank Wolfe Algorithm**
- The **COLA algorithm** is motivated by and related to the **Frank Wolfe algorithm**
- COLA is an iterative algorithm where each iteration is described by the equation: 
  $$W\leftarrow W+\arg\min_{BA}\mathcal{L}(W+BA)$$
- Taking a linear Taylor approximation, we can write:
  $$\mathcal{L}(W+BA)\approx L(W)+\nabla\mathcal{L}(W)BA$$
- This is reminiscent of the **Frank-Wolfe algorithm**, which was historically developed in the context of linear programming.

**Analyzing a Variant of the Frank Wolfe Algorithm**
- Below, we analyze a variant of the Frank Wolfe algorithm for stochastic non-convex smooth optimization
- The algorithm pseudo-code is given in **Algorithm 2**, written in COLA notations as an application to fine tuning of LLM
- The stochasticity is captured in equation (1), where it is assumed that the direction of the gradient is approximated up to ε using a stochastic gradient method.

**Idealized COLA Algorithm**
- The **idealized COLA algorithm** performs gradient updates such that after every epoch, Vt^T∇L(Wt)≤arg minW∈K{W^T∇L(Wt)}+ε

**Notation and Inequality**
- ht=𝄒(Wt)−𝄒(W*)
- gt=max⁡V∈K∇L(Wt)V−Wt

**Theorem 4.1**
- Algorithm 2 guarantees the following convergence guarantee for stochastic smooth nonconvex optimization:
  $$\frac{1}{T}\sum�b{1}g\_t\leq\frac{2√{Mβ}D}{\sqrt{T}}+\varepsilon$$
- Proof by induction, using the inequality ht+1≤ht+ηgt+ε+η²βD²/2
- The theorem holds as long as the distribution shift is bounded sublinearly with time.

## 5 Experimental Setup

**Table 1: Performance of OPT-1.3B on 1000 test examples across various tasks.**\n
- Averaged over five random seeds
- COLA outperforms LoRA in all tasks
- Tasks and models described first
- Comparison methods introduced
- Implementation details provided last

### 5.1 Models and Tasks

**Models Fine-Tuned:** Experimentation with COLA for fine-tuning OPT-1.3B [^32] and Llama2-7B [^26]. Both models' pre-trained checkpoints are from HuggingFace.

**Evaluation:** Assessment of our method against the LoRA baseline across seven classification tasks: SST-2, WSC, CB, WIC, BoolQ, MultiRC, and RTE for task adaptation effectiveness comparison.

**Methods Compared:** Primary focus on comparing with LoRA, a PEFT method for training low-rank matrices while maintaining frozen pre-trained model parameters. Future work will incorporate additional baselines.

### 5.2 implementation details

**Experimental Method**
- Implemented using **PyTorch** and **Transformers library** [^30]
- Conducted on **NVIDIA A100 (80G) GPU**
- Adopted experimental configuration from [^19]:
  - Randomly selected: 1000 examples for training, 500 for validation, and another 1000 for testing

**Training Details**
- **COLA**:
  - Used **AdamW** optimizer [^17] with 5 epochs
  - Applied **linear learning rate schedule** from: $\{1\times 10^{-3},8\times 10^{-4},5\times 10^{-4},1\times 10^{-4},5\times 10^{% -5}\}$
  - Set **batch size** to either 4 or 8
- **LoRA**:
  - Introduced **trainable linear low-rank modules** in query and value projections of all self-attention layers
  - Applied LoRA to specific matrices, not a pivotal aspect of the work [^31]
- **OPT Experiments**:
  - Incorporated bias into injected LoRA modules [^18]
- **Llama-2 Experiments**:
  - Disabled bias in LoRA to ensure key matching with pre-trained checkpoint "meta-llama/Llama-2-7b-hf"

**LoRA Configuration**
- Set the rank of LoRA (denoted as **"r"**) to 8 and **$\alpha$** to 16
- Used the ratio **$\alpha/r$** to scale weight updates

## 6 Results and analysis

### 6.1 Main Results

**Experimental Results**

**Test Performance of Method and Baseline**:
- Detailed in [Table 1](https://arxiv.org/html/2401.04151v1/#S5.T1 "Table 1 ‣ 5 Experimental Setup ‣ Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning") and [Table 2](https://arxiv.org/html/2401.04151v1/#S6.T2 "Table 2 ‣ 6.1 Main Results ‣ 6 Results and analysis ‣ Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning")
- Method consistently outperforms LoRA on all datasets within the same training budget

**Performance Boost for OPT-1.3B**:
- COLA brings a performance boost to LoRA by:
  - **WSC**: 3.66 (relative improvement of 6.47 $\%$ )
  - **BoolQ**: 1.38 (relative improvement of 1.95 $\%$ )
  - **RTE**: 1.66 (relative improvement of 2.29 $\%$ )

**Performance Boost for Llama2-7B**:
- COLA boosts the test score on RTE from 82.09 to 85.70, which corresponds to:
  - **Gain**: 3.61
  - **Relative improvement**: 4.40 $\%$

**Consistency and Training Budget**:
- Maintained consistency by setting the rank of all injected modules in the sequence to 8, aligning with the baseline LoRA setup
- Used an equal training epoch budget for different methods, ensuring the same training computation cost

### 6.2 Ablation Study

**COLAtuning with LoRA: Effects of Chain Length**

**Chain length in COLA:**
- Number indicating the number of LoRAs (Layer-wise Relative Adaptation) learned and merged during fine-tuning process.
- Longer chains result in better approximation for optimal weight update to pretrained language model (LLM).

**Experiments:**
- Conducted with varying chain lengths: 1, 2, 3.
- Total training epochs fixed at 5.
- Results reported over five random seeds.

**Findings:**
- Test accuracy increases as chain length grows.
- COLA more robust in terms of generalization error compared to baseline LoRA for most tasks.
- Lower standard deviations for COLA.

**Table 4:**
- Test scores and standard deviation reported for all tasks using 5 random seeds.
- Highest average performance highlighted in bold.

**Rank step-down:**
- Hypothesis: residual weight update for task adaptation decreases in rank.
- Studies conducted on COLA with length of two and varying ranks (2, 4, 6, 8) for remaining epochs.

**Results:**
- COLA with rank step-down outperforms fixed rank LoRA for all tasks except one data point.
- Superior generalization ability over standard LoRA and lower computational cost.
- Optimal rank is task-dependent.

**Figure 3:**
- Comparison of test performance for three tasks with COLA of length 2 and different ranks in the residual learning phase.
- Experiments conducted with fixed rank of 8 for first three epochs and varying ranks for remaining epochs.

**Computation comparison:**
- Training computation cost (FLOPs) provided for COLA of different configurations.
- Baseline LoRA uses a fixed rank of 8 throughout training, while COLA starts with rank 8 and continues with different ranks in the residual learning phase.
- Less compute required for COLA due to lower generalization error.

## 7 Conclusions and future work

**Chain of LoRA (COLA) for Fine-tuning Large Language Models**

Introducing COLA: An iterative low rank residual learning procedure to optimize weight updates for efficient task adaptation in large language models. Preliminary results demonstrate that COLA surpasses previous baselines while maintaining, or reducing, computational resources. Ongoing work involves testing COLA with various base optimizers and conducting larger scale experiments. Additionally, we are exploring applications beyond classification tasks, such as generation, summarization, and multiple choice tasks.

