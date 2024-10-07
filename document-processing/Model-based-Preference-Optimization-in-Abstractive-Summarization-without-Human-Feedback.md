# Model-based Preference Optimization in Abstractive Summarization without Human Feedback

by Jaepill Choi, Kyubyung Chae, Jiwoo Song Yohan, Jo Taesup Kim
https://arxiv.org/pdf/2409.18618

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Preliminaries](#2-preliminaries)
- [3 Proposed Method](#3-proposed-method)
- [4 Experiments](#4-experiments)
- [5 Analysis](#5-analysis)
- [6 Related Work](#6-related-work)
- [7 Conclusion](#7-conclusion)
- [A Appendix](#a-appendix)

## Abstract
**Abstractive Summarization:**
- Challenge: producing concise, accurate summaries from vast information
- Large Language Models (LLMs): generate fluent text but introduce inaccuracies
- Supervised fine-tuning methods: maximize likelihood but do not consistently enhance faithfulness of summaries
- Preference-based optimization methods: refine models to align with human preferences, but rely on costly human feedback

**Approach: Model-based Preference Optimization (MPO)**
- Novel approach to fine-tune LLMs for improved summarization abilities without human feedback
- Leverages model's inherent summarization capabilities
- Creates preference dataset using different decoding strategies
- Demonstrates enhanced quality of generated summaries in experiments on standard datasets and various metrics.
- Code available at https://github.com/cjaep/MPO.

## 1 Introduction

**1 Introduction:**
- Large Language Models (LLMs) have impressive capabilities in generating plausible text but may produce incorrect or contradictory information due to hallucination (Maynez et al., 2020).
- Reinforcement learning based objectives can help circumvent these failures by choosing appropriate reward functions (Paulus et al., 2018; Tian et al., 2024).
- Recently, reinforcement learning from human feedback (RLHF) has focused on aligning language models with human preferences to enhance summarization abilities (Böhm et al., 2019; Pasunuru and Bansal, 2018; Stiennon et al., 2020; Paulus et al., 2018; Ramamurthy et al., 2023).
- However, human feedback is not always reliable, as it may overlook factuality and consistency (Hosking et al., 2024). This implies preference optimization with human feedback does not guarantee improved faithfulness.
- Collecting high-quality human preference data is an expensive process (Min et al., 2023), leading to methods being proposed that do not rely on human feedback or external metrics for preference optimization.

**Challenges in Preference Optimization:**
- Relying excessively on imperfect metrics like ROUGE and FactScore carries the risk of overfitting (Strathern, 1997; Ramamurthy et al., 2023).
- Model-based Preference Optimization (MPO) proposed: a novel approach that leverages a model's inherent summarization capabilities without relying on any human feedback or external metrics.

**Methodology:**
- MPO uses deterministic decoding strategy (e.g., beam search decoding) to generate chosen samples and stochastic decoding strategy (e.g., temperature sampling) for rejected samples.
- Aligns model's preference toward summaries generated via beam search rather than randomly sampled ones based on findings that beam search yields more faithful summaries while randomness introduces hallucinations (Wan et al., 2023; Yang et al., 2018; Welleck et al., 2020; Holtzman et al., 2020).

**Results:**
- MPO outperforms models trained with standard supervised fine-tuning (SFT) or those optimized with human preferences (e.g., PPO, DPO) in terms of faithfulness and relevance to the source text, as shown in Figure 1.

**Conclusion:**
- Model-based Preference Optimization (MPO) is a simple and straightforward approach for fine-tuning language models to improve abstractive summarization without relying on any human feedback or external metrics. The experimental results demonstrate that MPO outperforms models optimized with human preferences, offering superior overall performance and greater generalizability across diverse language models and datasets.

## 2 Preliminaries

**Preliminaries**

**Problem Setup**:
- **Let V** denote the vocabulary for both input and output
- Represent input document as **x** and output summary as **y=⟨y0, ..., yT⟩ ∈ Y**
- Sequence **y** consists of T+1 elements, starting with **y0** (beginning-of-sequence token) and ending with **yT** (end-of-sequence token)

**Language Model (LM)**:
- Auto-regressive model of sequence distribution **P(y|x)**
- Each conditional probability is parameterized by a neural network **pθ**
- **P(y|x)** can be expressed as product of conditional probabilities:
  - **P(y|x) = TY t=1pθ(yt|y<t, x)**

**LM for Summarization**:
- Optimal summary **y∗** from set of valid strings **Y** is obtained using a scoring function:
  - **y∗ = argmax y∈Y pθ(y|x)**
- Finding optimal summary is not tractable, so scoring function for optimal string varies according to decoding strategies

**Decoding Strategies**:
- **Stochastic Decoding**:
  - Sample directly from probabilities predicted by model
    - **ytemp ∼ P(yt|x, y<t)**
  - Adjust variance using temperature parameter **τ**:
    - Increasing **τ** causes distribution to approach uniform, increasing risk of hallucinations
- **Deterministic Decoding**:
  - **Greedy Decoding**: Select most probable token at each step
    - **ygreedy = argmax y∈V logpθ(yt|y<t, x)**
  - **Beam Search Decoding**: Tracks top-k candidates and outputs best sequence among them
    - **ybeam = argmax y∈VLX t=1 logpθ(yt|y<t, x)**

**Choices for Preference Dataset**:
- Samples generated through stochastic decoding classified as rejected samples
- Samples generated via deterministic strategies (greedy or beam search) are chosen samples

## 3 Proposed Method

**Proposed Method for Encouraging Faithfulness in Abstractive Summarization**
- **Pipeline**: follows typical preference optimization pipelines (Rafailov et al., 2023; Ziegler et al., 2020; Sti-iennon et al., 2020; Ouyang et al., 2022)
- **Supervised Fine-Tuning (SFT)**: fine-tune pre-trained language model using supervised learning on training data `Dtrain={(x,yref)}`
	- Model trained to generate single sentence summary from source document
	- Utilize existing SFT models or apply SFT to pre-trained language models using QLoRA (Dettmers et al., 2023)

**Preference Optimization**:
- Employ **Direct Preference Optimization (DPO)**
- Simplifies process by eliminating need for explicit reward function, preferable to RL-based algorithms
- Model probability of observing preference pair using Bradley-Terry model: `p(yw≻yl) = σ(r(x,yw)-r(x,yl))`
	- Sigmoid function: `σ`
	- Reward function: `r(·,·)`
- Model directly learns policy from collected data without modeling reward function
- **DPO loss**: `LDPO(πθ; πref) = -E(x, yw, yl)~D log σ β log πθ(yw|x) πref(yw|x) - β log πθ(yl|x) πref(yl|x)`
	- Coefficient `β` controls trade-off between reward and divergence
	- Minimizes over-optimization (Tian et al., 2024)

**Constructing Preferences Pairs without Human Feedback**:
- Exploit differences between deterministic and stochastic decoding strategies
- **Dataset of preference pairs**: `Dvalid={(x, yw beam, yl temp)}`
	- Summaries generated through **beam search decoding** as chosen samples
	- Summaries from **temperature sampling** as rejected samples
- Conduct preference optimization with generated data to refine language model and ensure avoidance of hallucinated or irrelevant text.

## 4 Experiments

**TL;DR Dataset and eX-treme Summarization (XSUM) Dataset:**
* Widely used for abstractive summarization tasks
* TL;DR dataset: Reddit posts and their corresponding summaries
* XSUM dataset: BBC articles and their single-sentence summaries

**Models Used:**
* GPT-J (6B)
* Mistral-7B
* LLaMA2-7B

**Evaluation Metrics:**
* Faithfulness: Align Score and FactCC
* Relevance: BARTScore and BS-FACT
* Similarity: ROUGE-L

**Implementation Details:**
* SFT training with QLoRA: batch size 2, learning rate 1e-4, one epoch on train split
* Merging adapter into models for preference optimization experiments
* DPO hyperparameter βset to 0.5, learning rate 1e-4, and one epoch on validation split
* Limiting maximum generated tokens to 50 during summary generation
* Beam size of 6 for beam search decoding
* Temperatures: GPT-J (5.0), Mistral-7B (1.0), LLaMA2-7B (1.0)

**TL;DR Results:**
| Method            | Response Ratio | Faithfulness   | Relevance     | Similarity  | Align Score | FactCC      | BARTScore    | BS-FACT      | ROUGE-L     |
|------------------|---------------|--------------|--------------|-------------|------------|-------------|--------------|--------------|--------------|
| GPT-J (with ground-truth data)  | SFT          | 81.2% (99.4%) | 89.21 (83.54) | 64.18 (53.48) | -1.25 (-1.63) | 91.53 (90.30) | 26.74 (26.01) | 26.46 (25.75) | 64.25 (60.98) |
| GPT-J (with human feedback)    | PPO          | 100.0% (100.0%) | 83.10 (75.88) | 54.40 (47.52) | -1.35 (-1.80) | 91.32 (89.78) | 23.55 (23.28) | 23.36 (22.82) | 61.44 (60.89) |
| GPT-J (preference dataset)     | DPO          | 98.3 (99.8%) | 88.12 (82.55) | 61.70 (54.09) | -1.33 (-1.65) | 91.27 (90.22) | N/A         | N/A          | 60.58 (59.67)|
| Preferred-FT         | N/A         | 66.8% (99.6%) | 89.90 (82.04) | 76.58 (64.48) | -1.39 (-1.73) | 91.24 (90.09) | N/A         | N/A          | 65.64 (65.65)|
* MPO (Ours): 99.9% (99.9%) in both response ratio and all evaluation metrics.

**Comparing Methods:**
* Compared MPO with supervised fine-tuned models using ground-truth data or deterministic decoding
* Demonstrated that contrast between beam search decoding and random sampling is more effective than human-annotated preferences in terms of faithfulness.

### 4.2 Comparison with Fine-Tuned Models

**Comparison of MPO Model with Fine-Tuned Models**

**Table 1: Comparison of MPO with SFT, SFT++, Preferred-FT**
* MPO outperforms fine-tuned baselines (SFT, SFT++) in various metrics on the PO dataset
* SFT++ and Preferred-FT did not significantly improve over SFT
* Substantial increase in AlignScore (up to 3.28), FactCC (7.92), BARTScore (0.22), and BS-FACT (0.9) for MPO compared to SFT

**Table 2: Comparison of MPO with SFT**
* MPO demonstrates robust results across various language models on both TL;DR and XSUM datasets
* Lower ROUGE-L score compared to SFT, but not the primary focus
* MPO generally outperforms SFT in win rates when using different decoding strategies (greedy and beam search)

**Table 3: Comparison of MPO with Human Preference Optimized Models**
* MPO consistently outperforms human preference optimized models (PPO, DPO) based on automatic metrics
* Discrepancy in performance between the two groups in human evaluation
* Suggests AlignScore aligns with human judgement to some extent

**Table 4: Example Summaries of MPO Model and Human Preference Optimized Models**
* Superior summary generated by MPO compared to SFT and DPO (w/ human pref.) models in terms of faithfulness and source relevance

**Table 5: Results of Human Evaluation**
* MPO achieves an overall win rate of 51% compared to DPO based on GPT-3.5 evaluation
* MPO excels in faithfulness and source relevance, but may fall short in fluency
* MPO trained on fewer data pairs than human preference optimized models

**Table 6: Comparison of MPO with Decoding Strategies**
* Applicable to various decoding strategies (Nucleus, ITI, DoLa) despite not being specifically optimized for them
* Consistently produces enhanced summarization results compared to the standard SFT model in terms of faithfulness and relevance.

## 5 Analysis

**Evaluation of Chosen vs. Rejected Samples**
* Deterministic generation outperforms stochastic for summarization tasks (Table 7)
* Improving quality of rejected responses degrades performance (Table 8, Row 3)
* Similarity between deterministic and stochastic generated summaries is low (Table 9)
* Iterative MPO improves model's summarization ability by using previous iteration's beam search as chosen samples and random as rejected ones.

**Impact of Deterministic vs. Stochastic Generation on Summarization Performance:**
- Chosen samples: deterministic generation may lead to overly similar summaries, hindering the model's creativity (Table 9)
- Rejected samples: stochastic generation used as chosen samples can decrease faithfulness and relevance compared to original SFT model (Figure 3)

**Improving Model Performance through Iterative Training:**
- Using summaries generated via beam search from the previous iteration as chosen samples and random sampling from initial SFT model as rejected samples enhances dataset quality and improves model performance.

**Impact of Different Decoding Strategies on Summarization Performance:**
- MPO outperforms SFT in terms of AlignScore (Table 10)
- Incorporating more effective decoding strategies within MPO can further enhance summarization performance (comparison between MPO and Faithfulness-aware Lookahead, Table 10).

## 6 Related Work

**Related Work**

**Auto-Regressive Language Models**:
- Two primary approaches to enhance model's summarization capabilities:
  - Adjusting learning algorithm
  - Refining decoding strategy

**Decoding Strategies**:
- **Inference-time intervention (ITI)**: Shifts activations along truth-correlated directions
- **Decoding by contrasting layers (DoLa)**: Uses an early-exit strategy by contrasting differences in logits
- **Deterministic decoding strategy**(Wan et al., 2023): Extends the idea of lookahead and outperforms nucleus sampling in terms of faithfulness.

**Learning Algorithms**:
- **FactPegasus (Wan and Bansal, 2022)**: Employs a tailored pre-training setup with contrastive learning to generate more faithful summaries
- **RL-based objectives**: Can be used to enhance faithfulness (Böhm et al., 2019; Roit et al., 2023; Paulus et al., 2018)
  - RL provides a natural path for optimizing non-differentiable objectives in LM-based generation
  - **Direct Pref-erence Optimization (DPO)**(Rafailov et al., 2023) simplifies the process by eliminating the need for an explicit reward function of RL-based algorithms
  - Tian et al. (2024) suggested optimizing language models for factuality in long-form text generation using FactScore.

**Proposed Approach**:
- Trains the underlying model to provide summaries faithful to source documents, without requiring external metrics or human feedback during optimization process
- Versatile enough to integrate enhanced decoding techniques to more effectively reduce hallucinations.

## 7 Conclusion

**Model-based Preference Optimization (MPO) Approach:**
* Improves faithfulness and quality of abstractive summaries generated by Large Language Models (LLMs)
* Leverages model's inherent summarization capabilities to create preference dataset using different decoding strategies
* Extensive experiments demonstrate significant enhancement in summarization performance, providing an efficient solution for addressing challenges of hallucination in LLM-generated summaries.

**Limitations:**
* Employed QLoRA to maintain SFT model's performance but may have limited further improvements
* Absence of comparative experiments leaves uncertainty about actual effectiveness of QLoRA
* Limited experiments on 7B models raise concerns about scalability of approach
* Model increasingly adopts extraction approach during iterative training, posing challenge for more faithful abstractive summaries.

**Ethical Concerns:**
* Research relies extensively on outputs from self-supervised language models trained on unlabeled web-scale datasets which may learn and perpetuate social and ethical biases
* Use of TL;DR dataset, derived from Reddit posts, with potentially offensive content risks influencing model's outputs and further training within MPO, potentially perpetuating these biases.

**Acknowledgements:**
* Supported by National Research Foundation of Korea (NRF) grant and Institute of Information & Communications Technology Planning & Evaluation (IITP) grant, both funded by the Korean government.

## A Appendix

**GPT-3.5 Judgment Prompts**
- Used for evaluating win rates using GPT-3.5-turbo on prompts proposed by Rafailov et al., 2023
- Order of summaries or responses is randomly chosen for each evaluation
- Examples of prompts can be seen in **Figure 4**

**Human Evaluation Details**
- Sampled 100 instances from the TL;DR dataset for human evaluation
- Divided into four groups: A, B, C, and D based on AlignScore of DPO and MPO
- Goal was to assess reliability of Align Score in comparison to human judgment
- Asked annotators three questions: choose better summary (Q1), select summary with issues (Q2), mark problematic parts (Q3)
- Based on responses, confirmed method produced more faithful summaries than DPO in terms of consistency
- Despite similar win rates, DPO performed significantly worse in terms of consistency
- Evaluated 10 annotators, all STEM students from Seoul National University

**Example Cases and Statistics**
- Table **12**: Examples of different combinations of preference pairs
- Table **13**: Examples of summaries from iterative preference optimization
- All experiments conducted on NVIDIA GPUs, with varying time requirements
- Parameters for package: ROUGE and BERTScore loaded from evaluate package version 0.4.1

**Text Source** (for evaluation examples)
- TL;DR dataset: constructed by Reddit posts and their corresponding summaries
- XSUM dataset: BBC articles and their corresponding summaries

**Analysis on Error Bars, Reproducibility, Parameters for Package, and Iterative Preference Optimization**
- Details not provided in the text.

