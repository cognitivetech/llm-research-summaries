# HunyuanProver A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving

source: https://arxiv.org/html/2412.20735v1
Yang Li, Dong Du\*, Linfeng Song\*, Chen Li\*, Weikang Wang, Tao Yang and Haitao Mi  

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Scalable Data Generation for Prover Improving](#2-scalable-data-generation-for-prover-improving)
  - [2.1 Autoformalization Data Generation](#21-autoformalization-data-generation)
  - [2.2 Prover Improving via Iterative Tactic-Level Proving Data Generation](#22-prover-improving-via-iterative-tactic-level-proving-data-generation)
- [3 Guided Tree Search](#3-guided-tree-search)
  - [3.1 Tree Search Algorithms](#31-tree-search-algorithms)
  - [3.2 Critics for Search Guidance](#32-critics-for-search-guidance)
- [4 Experiments](#4-experiments)
  - [4.1 Setup](#41-setup)
  - [4.2 Results and Analysis](#42-results-and-analysis)
- [5 Conclusions](#5-conclusions)
- [Appendix A Examples Theorems Proved by HunyuanProver](#appendix-a-examples-theorems-proved-by-hunyuanprover)

## Abstract

We introduce HunyuanProver, a language model fine-tuned from Hunyuan 7B for interactive theorem proving with LEAN4. To address data sparsity, we designed an iterative synthesis framework and guided tree search algorithms to facilitate "system 2 thinking". HunyuanProver achieves state-of-the-art performance on major benchmarks, including a 68.4% pass rate on miniF2F-test. We also open-source a dataset of 30k synthesized instances for the community's benefit.

## 1 Introduction

**Recent Advancements in Large Language Models (LLMs) and Formal Theorem Proving:**
- LLMs have significantly impacted mathematical reasoning and theorem proving in AI
- Notable progress in natural language domains, but challenges remain in formal theorem proving using systems like LEAN or Isabelle
- Even advanced models like GPT-4 struggle with complex formal statement proving due to large search space and limited data

**Challenges in Automatic Theorem Proving:**
- Understanding both syntax and semantics of a formal system for generating valid next steps
- Integrating abstract mathematical reasoning skills for effective and efficient deductions

**Proposed Solution: HunyuanProver Framework**
- Consists of two core modules: scalable prover-data generator and guided tree-search algorithms
- Prover-data generator uses open-source data to train the initial autoformalizer and prover
- Autoformalizer converts existing math questions into target prover format (e.g., LEAN4) for iterative improvement
- Tree-search algorithms and multiple critic models perform "slow thinking" for solving complex theorem proving tasks

**Evaluations:**
- Achieves an accuracy of 68.4% on the miniF2F benchmark
- Key findings:
  - Using explicitly trained critic for tree-search guidance is helpful
  - Scale of finetuning data for theorem proving is critical, and designing efficient data scaling framework is important
  - Data curation and selection are crucial when sufficient training data exists

**Key Contributions:**
- Introduce a scalable pipeline using open-source prover data and natural language math problems to generate new training data for automatic theorem proving
- Develop several critic models and evaluate their effectiveness with best-first search and Monte Carlo tree search algorithms
- Will open-source approximately 30,000 data instances, including formal statements, proofs searched by HunyuanProver, and original problems.

## 2 Scalable Data Generation for Prover Improving

**Automated Theorem Proving Challenges**
- **Lack of training data**:
  * Limited availability of large open-source datasets (e.g., mathlib4 [^6] containing around 50k theorems)
  * Insufficient for training stronger provers due to the difficulty of automatic theorem proving

**Scaling Training Data for Automated Theorem Proving**

**Autoformalization**:
- Maps natural language problems into LEAN format statements

**Tactic Generation**:
- Supports iterative theorem proving by generating tactics

**Prover Improving**:
- Section [2.1] (Autoformalization): Data Generation
  * Scalable method for autoformalizing natural language problems into LEAN format statements
- Section [2.2] (Prover Improving): Iterative Tactic-Level Proving Data Generation
  * Generates tactics for iterative theorem proving to improve prover performance.

### 2.1 Autoformalization Data Generation

We start with 130k high-quality LEAN statement pairs in natural language, including 50k from Lean workbook and 80k from MMA. We translate these to Chinese, double the dataset size, and train an autoformalization model. This model converts internal math problems (30 million) into formal statements. After filtering out non-conforming outputs, we get a dataset D^q of 20 million LEAN statements.

### 2.2 Prover Improving via Iterative Tactic-Level Proving Data Generation

**Iterative Framework for Generating Tactic Data**

**Designing an Iterative Framework**:
- Takes a LEAN engine Γ and statement dataset D^q from autoformalization to generate new tactic data for prover improvement.
- In iteration t, leverages best-first search (BFS) algorithm with the prover from the previous iteration π_t-1 on all unsolved statements in D.
- Collects proving trajectories (e.g., τ) for newly solved statements q if there is any: D_t = {(q,τ) | q ∈ D^q - D_{t-1}, τ ∼ Bfs(q), τ ≠ null} ∪ D_{t-1}.
- Updates the prover using rejection finetuning (RFT) with the proving trajectories in D_t after filtering out easy statements solved in early iterations.
- The initial prover π_0 is trained on public data including mathlib4.
- After more than 10 rounds of iteration, more than 20 million tactic-level data is obtained.

**Enhancing Diversity**:
- **Prover diversity** is important due to the massive search space of theorem proving.
- Two methods are developed to enhance prover diversity within the iterative data-generation framework:
  1. Design rules to convert the last state of an unfinished proving trajectory into a new statement, obtaining more diverse proving data for prover training.
  2. Collect data from proving more challenging statements, including Olympiad-level algebraic inequalities and Lean workbook problems.

**Figure 1**: Comparison of Best-First Search (BFS) and Monte-Carlo Tree Search (MCTS):
- BFS only takes **Selection** and **Expansion** in one iteration, while MCTS takes all four steps.
- The numbers represent critic-assigned scores.

## 3 Guided Tree Search

**HunyuanProver: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving**

**Process Overview**:
- Iterative interaction with LEAN environment
- Policy predicts new tactic given a state in proving process
- Abstracted as tree search
  - State `s_i` corresponds to node `n_i`
  - Edge from `n_i` to `n_j` represents applying a tactic on `n_i` to yield `n_j`

**Tree Search Algorithms**:
- Two major algorithms designed, as described in Section [3.1](https://arxiv.org/html/2412.20735v1#S3.SS1 "3.1 Tree Search Algorithms ‣ 3 Guided Tree Search ‣ HunyuanProver: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving")

**Critics for Search Guidance**:
- Several critics designed to guide these algorithms, as shown in Section [3.2](https://arxiv.org/html/2412.20735v1#S3.SS2 "3.2 Critics for Search Guidance ‣ 3 Guided Tree Search ‣ HunyuanProver: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving")

### 3.1 Tree Search Algorithms

**Best-First Search (BFS)**
* Selection step:
  * Select node n\_hat with highest critic score from active nodes set 𝒩: n\_hat = arg max\_nin𝒩 Critic(n)
* Expansion step:
  * Sample K candidate tactics under n\_hat
  * Execute each tactic against LEAN engine, yielding new tree node if valid
  * Merge remaining nodes into active node set 𝒩 after removing identical nodes
* Limitations:
  * Each node visited only once with fixed expansion budget
  * Relies solely on critic model's guidance and may suffer from biases or misjudgments

**Monte-Carlo Tree Search (MCTS)**
* Original algorithm takes selection, expansion, simulation, and back-propagation steps
* Removed simulation step for future work
* Follows BFS setting to sample K candidate tactics at a time instead of one original MCTS
* Differences from BFS:
  * Node can be sampled multiple times with dynamic importance score
  * Expansion budget updated by importance score: E(n) = max(B\_min, min(B\_max, ⌊α I(n)⌋+1))
  * Selection based on Upper Confidence Bound (UCB): UCB(n) = Critic(n) + α√(2xlnCnt(Prnt(n))/Cnt(n))
    * Balances exploitation (high critic score) and exploration (under-explored nodes)

### 3.2 Critics for Search Guidance

**Guided Tree Search: Critic Modeling Approach**

**Policy Confidence (PC)**
- Provides guidance for cold start of guided search due to limited tree-search data
- Defined as token-level average log probability: `f^π(c) = 1/|c| * sum_j=1^|c|(log p_π(c^j|q,n,c^<j))`
- Where `|c|` is the number of tokens in `c`, and `p_π(c^j|q,n,c^<j)` is the policy probability

**Process Reward Model (PRM)**
- Represents possibility of proving statement `q` from tree node `n` under policy `π`
- Trained using a search tree and node scores assigned by human experts or approximated
- Minimizes mean squared error between predicted and actual node scores: `v_ϕ^π = -E_(q,n,l) * (v_ϕ^π(q,n) - l)^2`
- Outputs a scalar prediction at the last token of each state as value

**Distance Critic (DC)**
- Predicts estimated remaining number of steps to prove `q` from `n`
- Trained to identify path on balanced binary tree representing numbers from 1 to 8
- Helps mitigate data sparsity by enabling coarse-to-fine predictions
- Each node on the tree is represented by a special token, such as `<|num-1-of-2|>` for 1/2
- During search stage, states are compared by directly comparing their corresponding tuples.

## 4 Experiments

### 4.1 Setup

**Benchmarks for Theorem Proving**

**MiniF2F Benchmark**:
- Examines LLM's automatic formal problem-solving skills
- Focuses on high-school level exercises and competitions, e.g., AMC, AIME, IMO
- Includes 244 validation problems and 244 test problems in algebra and number theory

**Inference**:
- Uses LEAN engine from LeanDojo as:
  - Tactic data generator
  - Benchmark evaluator
- Sets timeout limits for tactic execution:
  - Whole: 3600 seconds
  - Per step: 60 seconds
- Limits search steps to 800 for both BFS and MCTS
- Samples 2 tactics under each temperature (0.7, 0.8, 1.0, 1.1) from the given LEAN state

**Finetuning Hyperparameters**:
- Prover obtained by fine-tuning a Hunyuan 7B model on self-generated tactic data
- During fine-tuning:
  - Conducted at most 4 epochs
  - Checkpoint selected based on miniF2F valid set
- Set parameters:
  - Maximum sequence length: 4096
  - Learning rate: 2x10^-5
  - Minimal learning rate: 1x10^-6
  - Batch size: 256
  - Cosine learning schedule used

**Comparing Systems**:
- Compares HunyuanProver with former state-of-the-art systems:
  - Lean-STaR and InternLM2.5-StepProver (interactive, step-level proving methods)
  - DeepSeek-Prover-V1.5 (whole-proof generation method)

### 4.2 Results and Analysis

**Comparison of HunyuanProver with Other Systems (MiniF2F-test)**
* CG: critic-guided search
* DC: distance critic as guidance
* Cost representation for BFS methods: #Pass x #Beam x #Iteration, for MCTS: #Pass x #Iteration

**Table 1 Comparison**
| System | Accuracy on miniF2F-test | Sampling Cost |
|---|---|---|
| HunyuanProver (CG+DC) | **Best** | N/A |
| InternLM2.5-StepProver+BFS+CG | Slightly better than DeepSeek-Prover-V1.5-RL+MCTS | N/A |
| DeepSeek-Prover-V1.5-RL+MCTS | N/A | Higher |
| HunyuanProver (BFS+CG) | Second best | Lower |

**Performance Boosts through Iterative Tactic Data Generation**
* Performance boosts in early iterations until version v8
* Minor improvements from v8 to v12 with increased training tokens
* Removal of easy data after v12 leads to performance boost (Figure 3)
* Importance of data selection during the iterative improving process

**Effectiveness of Different Critics and Tree Search Methods**
* HunyuanProver v16 vs. v16+DC: more deep proofs found in miniF2F for v16+DC (Figure 4)
* PRM and DC are better choices than policy confidence as they only require prover-generated data with natural labels.
* Table 2 ablation study: MCTS-with-PRM consistently better than BFS with policy confidence, DC significantly effective.

## 5 Conclusions

We present HunyuanProver, a system that boosts automatic theorem-proving in LEAN through iterative tactic data generation and guided tree search. The method significantly improves performance by expanding the training dataset 40-fold. Critic-guided algorithms further enhance effectiveness. Future work includes refining prover training data and exploring alternative cost-efficient tree search methods like Q*.

## Appendix A Examples Theorems Proved by HunyuanProver

**Proof Outline:**
* **Case for neg x < 2**:
  + Use lemma: Real.rpow_le_rpow_left_iff
  + By norm_num as by
* **Case for x >= 0**:
  + Use lemma: Real.rpow_one and Real.rpow_mul
  + x^2 <= 1 iff x <= 1 (squared numbers)
  + x^2 <= 1 iff x <= sqrt(1) && x <= 1/x (bounded by 1 and 1/x)
* **Conclusions:**
  + All goals proved using linear arithmetic, Aesop, and given lemmas.

