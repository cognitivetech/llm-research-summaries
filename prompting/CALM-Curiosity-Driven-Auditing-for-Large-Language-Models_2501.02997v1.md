# CALM Curiosity-Driven Auditing for Large Language Models

source: https://arxiv.org/html/2501.02997v1
bt Xiang Zheng, Longxiang Wang, Yi Liu, Xingjun Ma, Chao Shen, Cong Wang

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Preliminaries](#3-preliminaries)
- [4 Curiosity-Driven Auditing](#4-curiosity-driven-auditing)
  - [4.1 Regularized Auditing Objective](#41-regularized-auditing-objective)
  - [4.2 Token-Level Intrinsic Bonus](#42-token-level-intrinsic-bonus)
- [5 Experiments](#5-experiments)
  - [5.1 Experiments Setup](#51-experiments-setup)
  - [5.2 Inverse Suffix Generation](#52-inverse-suffix-generation)
  - [5.3 Toxic Completion Task](#53-toxic-completion-task)
- [6 Conclusion](#6-conclusion)

## Abstract

**Auditing Large Language Models (LLMs)**
- Crucial and challenging task
- Focus: auditing black-box LLMs without access to parameters, only service
- Goal: uncover input-output pairs exhibiting illegal, immoral, or unsafe behaviors
  - Example: non-toxic input leading to toxic output or hallucinative response containing politically sensitive individuals
- Challenges:
  - Scarcity of feasible points
  - Discrete nature of prompt space
  - Large search space

**Proposed Solution: Curiosity-Driven Auditing for Large Language Models (CALM)**
- Uses intrinsically motivated reinforcement learning to finetune LLM as auditor agent
  - Identifies potential harmful and biased input-output pairs of target LLM
- Successfully identifies derogatory completions involving celebrities and uncovers inputs eliciting specific names under black-box setting

**Important Details:**
- Code available at https://github.com/x-zheng16/CALM.git

**Content Warning:**
- This paper includes examples that may be offensive.

## 1 Introduction

**Large Language Models (LLMs)**
- Significant advancement in artificial intelligence
- Produce human-like text with impressive fluency and understanding of context
- Wide-ranging applications: natural language comprehension, creative content generation
- Importance in education, industry, and research

**Challenges with LLMs**
- Potential to generate toxic or hallucinative outputs
- Complex internal processes make safe use a concern
- Auditing is essential for managing risks

**Auditing Objectives**
- Identify undesired behaviors: toxic content, stereotypes, discrimination, leaking private information
- Multivariate function r(𝐬, 𝐨) captures specific auditing objectives
- Focus on two objectives: generating specific suffixes and toxic completions about celebrities

**Auditing Methods**
- Primarily relied on manually created prompts with limitations in exploring model input space and detecting rare harmful outputs
- Research shows harmful behaviors can be context-dependent, challenging traditional auditing methods to uncover infrequent cases

**Examples of Audit Tasks**
| Target LLM | Input of Audit LLM | Output of Audit LLM | Input of Target LLM | Output of Target LLM |
| --- | --- | --- | --- | --- |
| GPT-2 | Ask questions on senator facts (e.g., Florida, Delaware) | Induces sensitive information | Senators' records or accusations | Specific names and sensitive/offensive information |
| Dolly-2-7B |  |  |  |  |
| Llama-2-7B | Ask questions on senator facts (e.g., Arkansas, Colorado) | Induces sensitive information | Senators' records or accusations | Specific names and sensitive/offensive information |
| Llama-3-8B | Ask questions on senator facts (e.g., Michigan, Vermont) | Induces sensitive information | Senators' records or accusations | Specific names and sensitive/offensive information |

## 2 Related Work

**Algorithmic Auditing**
- Crucial for ensuring development and deployment of artificial intelligence systems, especially complex models like Large Language Models (LLMs) operating in high-stakes environments
- Involves systematically evaluating a model's behavior to ensure ethical and safety standards, identify potential biases, and assess compliance with legal and regulatory requirements
- Traditional auditing methods rely on static datasets and predefined benchmarks, which may not capture the full range of behaviors in complex models like LLMs
- Recent work emphasizes the importance of **dynamic and adaptive auditing strategies** to explore model's behaviors and uncover hidden risks effectively

**LLM-assisted Red Teaming**
- Proactive method for stress-testing black-box AI systems, such as LLMs, by simulating adversarial scenarios with a red-team LLM to find weaknesses of the target LLM
- Unlike traditional red teaming techniques involving human adversaries manually testing the system, LLM-assisted methods leverage pre-trained LLMs to automate the process
- Red-team LLM is instructed to generate diverse adversarial inputs
- Effective in identifying **edge cases and failure modes** that may not be found through conventional testing or fuzzing methods.

## 3 Preliminaries

**Components of CALM**

**Interaction with the Target LLM:**
- Model target LLM as a stochastic black box function that generates outputs based on user prompt 𝐬_T
- Denoted as f(·|𝐬), reflects top-k or top-p decoding strategies
- Goal: identify input-output pairs where output exhibits undesirable behaviors without access to internal parameters

**Reinforcement fine-tuning of the Audit LLM:**
- Model token generation as a partially observable Markov Decision Process (POMDP)
- Each token generation is treated as an action, previous tokens constitute observable state
- Denoted as π, tunable audit LLM predicts next token based on initial prompt 𝐳 and sequence of previously generated tokens 𝐬_t-1
- At each step t:
  - s_t sampled via s_t ~ π(·|𝐳) at step one
  - Next token generated as s_2 ~ π(·|𝐳,[s_1]) at step two
- Utilize modern RL algorithms like Proximal Policy Optimization (PPO) to fine-tune audit LLM by maximizing expected rewards.

## 4 Curiosity-Driven Auditing

**Problems of Previous Auditing Methods**
- **White-box methods**: require full access to model's internal parameters, impractical for cloud-deployed models
- **Zero-order gradient**: computationally expensive and often infeasible for LLMs in black-box settings
- **Hand-crafted prompts**: 
    - Require extensive expert knowledge
    - Labor-intensive to create
    - Narrow scope, leaving many harmful behaviors undetected

**Approach**
- Propose finetuning an audit LLM via intrinsically motivated reinforce learning (RL)
- Finetune audit LLM to automate prompt generation for target LLM
- Use novel regularized auditing objective:
    - Primary auditing objective
    - Intrinsic objective as regulator
- Design curiosity-driven exploration bonuses based on policy cover theory

### 4.1 Regularized Auditing Objective

**CALM's Approach to Exploring Input Space for Harmful Behaviors:**
- **Intrinsically Motivated RL**: CALM uses intrinsic motivation for fine-tuning the audit LLM
- **Audit LLM as an Agent**: Acts as a reward-based agent, maximizing extrinsic and intrinsic rewards
- **Extrinsic Reward**: Detecting harmful output behaviors (r(𝐬,𝐨))
- **Intrinsic Reward**: Encourages exploration in token embedding space (r^E(s))
- **Optimization Objective**: Regularized auditing objective with extrinsic, intrinsic rewards and KL penalty
  * Extrinsic: J_A(𝐬)
  * Intrinsic: J_I(s)
  * KL divergence term: J_KL(s)
- **Hyperparameters**: λ_I and λ_KL control trade-offs between objectives

**Auditing Objectives:**
- **Inverse Suffix Generation**: Audit LLM generates suffixes to evoke specific celebrity names (r(𝐬,𝐨))
- **Toxic Completion**: Generates adversarial prompts for toxic content about celebrities (NonToxic(𝐬) & Toxic(𝐨))

**Audit LLM's Performance:**
- Induces prompt distribution: P_𝐬^π
- Induces token distribution: P_s^π with discount factor γ
- Extrinsic objective: J_A([𝐬,𝐨]) = expected reward based on target LLM's response under the induced prompt distribution
- Intrinsic objective: J_I(s) = expected intrinsic bonus measuring novelty of token in token embedding space (mathcal{T})

### 4.2 Token-Level Intrinsic Bonus

**Design Rationale for Intrinsic Bonus:**
* Measures novelty of state
* Various intrinsic motivation techniques: knowledge-based vs data-based
	+ Knowledge-based estimates all historical experiences
	+ Data-based only concerns current experience
* Adopted technique: policy cover theory

**Policy Cover Theory:**
* Estimates the weighted sum of all historical token distributions
* Encourages agent to explore novel regions in prompt space
* Intrinsic objective: maximize deviation from policy cover

**Designing Token-Level Intrinsic Bonus R_I(s):**
* Based on policy cover theory
* Leverage concept of ρ(s) as estimated historical token distributions
* Maximize deviation between current policy and policy cover

**Practical Implementation:**
* Approximate inverse of policy cover using prediction error of a random neural network
* Final intrinsic bonus: R̂_I(s) = ψ_1(h) - g_1(h)ψ_2(h) - g_2(h)

**Auditing Process:**
1. Initialize audit LLM, value function, step counter, policy update step counter, total policy updates, prompt length, output length, auditing objective, and initial prompt set for the audit LLM according to the audit task.
2. Collect samples with s_t ~ π_θ_l(·|𝐳,𝐬_t-1) and 𝐨_N ~ f(·|𝐬_T).
3. Compute auditing reward r(𝐬,𝐨) via Equation 2 or 3.
4. Compute intrinsic bonus R̂_I(s) via Equation 6 and policy loss L_θ using PPO.
5. Update audit LLM's parameters θ and value function V(𝐬_i) via stochastic gradient ascent on L_θ.
6. Repeat until TotalSteps are reached.

## 5 Experiments

**Experiments on CALM (Contrastive Adversarial Latency Model)**

**Performance in Inverse Suffix Generation Task:**
- Figure 1: Performance with intrinsic coefficient lambda=10 (source: arxiv.org)
  * Figure shows results of a study assessing the effectiveness of CALM in this specific task

**L0 Norm of NameSet Coverage:**
- Figure 2: L0 norm analysis in inverse suffix generation task with lambda=10 (source: arxiv.org)
  * Provides additional information about the performance improvements from CALM in this task

**Ablation Study on Intrinsic Coefficient:**
- Figure 3: Ablation study results for intrinsic coefficient lambda=100 (source: arxiv.org)
  * Offers insights into how changes to the intrinsic coefficient affect the performance of CALM in inverse suffix generation task

**Performance in Toxic Completion Task:**
- Figure 4: Results from toxic completion task with intrinsic coefficient lambda=10 (source: arxiv.org)
  * Demonstrates how CALM can reveal undesirable outputs from target LLMs even when model parameters are inaccessible.

### 5.1 Experiments Setup

**Experimental Setup**
- **Audit LLM Backbone**: GPT-2 fine-tuned last two transformer blocks for adaptability and computational efficiency
  * Lightweight, essential text generation ability
  * Balances adaptability and computational power
- **RL Backbone**: PPO (Proximal Policy Optimization) algorithm used for reinforcement fine-tuning of audit LLM
  * On-policy RL algorithm
  * Runs on Nvidia A6000 GPU (48G) for handling high dimensionality

**Toxicity Classifier Implementation**
- **Approach**: Simple toxicity classifier checking for Not-Safe-For-Work (NSFW) words
  * Transparent and less prone to adversarial attacks
  * Directly checks for specific problematic terms
  - Effective for detecting overtly toxic language reliably
  - Based on well-established criteria from previous research

**Baseline Selection**
- **Baselines**: RL (Reinforcement Learning) and CRT (Contrastive Reinforcement Training) methods adopted as baselines
  * Justification for selection: Appendix C

### 5.2 Inverse Suffix Generation

**Comparison of Auditing Methods: Inverse Suffix Generation Task**

**Performance of the audit LLM**:
- [Figure 1](https://arxiv.org/html/2501.02997v1#S5.F1) illustrates convergence behavior of audit LLM when auditing various target black-box LLMs: GPT-2, Dolly-2-7B, Llama-2-7B, and Llama-3-8B
- Both CALM and RL methods converge towards auditing objective as number of queries increases
- Indicates effective adaptation to task, improving performance over time and generating desired suffixes

**Performance Analysis: L0 Norm of NameSet Coverage**:
- [Figure 2](https://arxiv.org/html/2501.02997v1#S5.F2) measures well each method covers desired set of names during generation process
- Observation: CALM exhibits lower variance compared to RL methods, especially for complex models like Llama-3-8B
  - Suggests more stable and reliable results for effective auditing

**Ablation Study on Intrinsic Rewards**:
- Analyzes effect of intrinsic rewards on audit LLM performance when auditing Llama-3-8B model in inverse suffix generation task with larger intrinsic coefficient λ=100
- [Figure 3](https://arxiv.org/html/2501.02997v1#S5.F3) illustrates three metrics: Auditing Objective, L0 Norm of Set Coverage, and Entropy of Set Coverage

**Auditing Objective**:
- Gradual improvement over time, suggesting enhancement in model's capacity to explore large token embedding space

**L0 Norm of Set Coverage**:
- Rapid convergence signifies intrinsic rewards' efficacy in guiding efficient exploration of desired output space

**Entropy of Token Distribution**:
- Initially high, indicating diverse possible outputs
- Gradually decreases over time, suggesting model focuses on specific outputs without sacrificing diversity.

### 5.3 Toxic Completion Task

**Toxic Completion Task Results for CALM:**
* **CALM outperforms baseline methods (RL and CRT)** in [Figure 4](https://arxiv.org/html/2501.02997v1#S5.F4) across all tested models in the senator-related toxic completion task:
	+ Significant margins of over 35% and 50% improvement for GPT-2 and LLAMA3 models, respectively
	+ Baseline methods exhibit lower peak performance and do not reach efficacy of CALM
	+ Sentence-level diversity score in CRT detrimentally impacts the performance of vanilla PPO, highlighting importance of token-level intrinsic bonus for enhancing audit efficacy
* **CALM demonstrates faster convergence:**
	+ Achieves over 80% auditing objective for Llama-3-8B with approximately 1.5x10^4 queries
	+ Reaches 50% accuracy rate with just 1x10^4 queries, significantly faster than baseline methods
* **CALM exhibits greater stability:**
	+ Consistently lower variance in results compared to RL and CRT, which are prone to more pronounced fluctuations

**Limitations:**
* Current study adopts lightweight GPT-2 as audit LLM backbone for CALM
* Belief that more powerful auditor backbone will enhance CALM’s performance.

## 6 Conclusion

We proposed CALM, a system that fine-tunes an audit Large Language Model (LLM) using intrinsic motivation to identify harmful and biased input-output pairs in black-box LLMs. CALM outperformed existing methods, successfully detecting toxic completions and uncovering concerning inputs from target models.

