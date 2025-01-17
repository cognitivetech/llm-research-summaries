# Extracting Unlearned Information from LLMs with Activation Steering

Article Source: arXiv.org ([2411.02631v1](https://arxiv.org/html/2411.02631v1))
Authors: Atakan Seyitoğlu, Aleksei Kuvshinov, Leo Schwinn, Stephan Günnemann
Affiliation: Department of Computer Science & Munich Data Science Institute, Technical University of Munich
Email: {a.seyitoglu, a.kuvshinov, l.schwinn, s.guennemann}@tum.de

## Content
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Anonymized Activation (AnonAct) Steering](#3-anonymized-activation-anonact-steering)
  - [3.1 Problem Description](#31-problem-description)
  - [3.2 AnonAct](#32-anonact)
- [4 Experiments](#4-experiments)
  - [4.1 Models](#41-models)
  - [4.2 Datasets](#42-datasets)
  - [4.3 Experiment Setting](#43-experiment-setting)
- [5 Results \& Discussion](#5-results--discussion)
- [6 Conclusion](#6-conclusion)

## Abstract

**Unintended Consequences of Large Language Models (LLMs)**
- Vast pretraining leads to:
  - **Verbatim memorization** of training data fragments
  - Potential containment of sensitive or copyrighted information

**Addressing Sensitive Knowledge in LLMs**
- **Unlearning** has emerged as a solution to remove sensitive knowledge
- Recent work shows supposedly deleted information can still be extracted:
  - Through various attacks
  - Retrieves sets of possible candidate generations, unable to pinpoint actual target information

**Proposed Solution: Activation Steering**
- Method for **exact** information retrieval from unlearned LLMs
- Introducing a novel approach to generating steering vectors, named **Anonymized Activation Steering**
- Developing a simple word frequency method to pinpoint the **correct** answer among a set of candidates when retrieving unlearned information

**Evaluation Results**
- Across multiple unlearning techniques and datasets:
  - Successfully recovers general knowledge (e.g., widely known fictional characters)
  - Reveals limitations in retrieving specific information (e.g., details about non-public individuals)
- Demonstrates **exact** information retrieval from unlearned models is possible, highlighting a severe vulnerability of current unlearning techniques.

## 1 Introduction

**Large Language Models (LLMs)**
- Trained on vast amounts of text data from diverse sources
- Generate high-quality responses to a wide range of topics

**Challenges with LLMs**
- Verbatim memorization of fragments of training data
- May contain sensitive information, such as:
  - Personal addresses
  - Passages from copyrighted books
  - Proprietary code snippets

**Regulations and Unlearning**
- Regulations like GDPR aim to give individuals control over personal information
- Retraining LLMs from scratch is impractical, so unlearning is an alternative solution
- Unlearning methods aim to remove knowledge while preserving overall performance

**Evaluating Unlearning Methods**
- Difficult to assess if unlearned concepts are fully forgotten
- Existing approaches recover unlearned information but cannot pinpoint the answer containing the correct information

**Contributions of this Paper**
- Introduce a novel attack based on **activation steering** against LLM unlearning
  - Deploying a novel way of generating pairs of prompts to calculate activation differences: **Anonymized Activation Steering**
- Evaluate the approach on three unlearning methods and three different datasets
  - Demonstrate that it can retrieve unlearned information in an **exact** manner, pinpointing the correct answer among a set of candidates with high accuracy
- Investigate failure scenarios and find that exact information retrieval is successful for general knowledge but fails for specific, less known information
- Provide a new dataset for Harry Potter information retrieval to enable more accurate assessments of unlearning methods

## 2 Related Work

**LLM Unlearning**:
- Efficient research topic compared to retraining from scratch
- Methods exist for unlearning entire domains of information (e.g., Harry Potter franchise) [^3]
- Enables deletion of personal information at request [^4]
- Replacing specific information possible (unlearning a single sentence) [^10]

**Benchmarking Unlearning Methods**:
- TOFU dataset: Synthetic data about fictitious authors for direct comparison between unlearned and baseline models [^6, 10]
- WMDP dataset: Designed specifically for benchmarking unlearning techniques, contains harmful information [^6]
- Attacks on standard models to recover information (jailbreak setting) [^2]

**Unlearning Methods and Challenges**:
- ROME and MEMIT methods prone to attacks, information not truly deleted [^11, 12]
- Activation steering technique for manipulating LLMs' latent space and guiding responses [^1, 18]
- Overcomes refused prompts, reduces toxicity, enhances truthfulness, adjusts tone of responses [^1, 5, 20, 22]
- Calculates steering vector from target direction for guiding model output [^18]

## 3 Anonymized Activation (AnonAct) Steering

### 3.1 Problem Description

**Unlearning Model: Extracting Information**
- **Ask straightforward questions**: contain specific keywords as answers
- **Determine model's response accuracy**: without prior knowledge of unlearning topic (finetuning)
- For base, non-unlearned model:
  - Determining correct answer is trivial if necessary knowledge is present
- For unlearned model:
  - Information leakage occasionally makes correct answer appear among responses
  - Correct answer frequency (CAF) is low compared to base model

**Anonymized Activation (AnonAct) Steering**:
- Proposed method to increase CAF and effectively undo unlearning process

### 3.2 AnonAct

**Anonymized Activation (AnonAct) Steering Method**

**Anonymization Strategy**:
- Generate multiple anonymized versions of a question Q
- Anonymize simple entities manually, complex terms using large language model (GPT-4)
- Ensure anonymized questions are not related to the unlearned domain

**Extraction of Steering Vectors**:
- Follow established activation steering methods from literature [^1]
- Extract internal representations between layers of the Large Language Model (LLM) for each anonymized question
- Compute difference between these representations and those of the original questions
- Average differences across all anonymized questions to obtain steering vector for that layer

**Application of Steering Vector**:
- Add steering vector back during generation with scaling factor, only for first token
- Limit application of steering vector to first token as it significantly impacts the entire sequence [^25]

**Figure 1: Anonymization Example**:
![Refer to caption](https://arxiv.org/html/2411.02631v1#S3.F1 "Figure 1 ‣ 3.2 AnonAct ‣ 3 Anonymized Activation (AnonAct) Steering ‣ Extracting Unlearned Information from LLMs with Activation Steering")

**Figure 2: Visual representation for AnonAct**:
![Refer to caption](https://arxiv.org/html/2411.02631v1#S3.F2 "Figure 2 ‣ 3.2 AnonAct ‣ 3 Anonymized Activation (AnonAct) Steering ‣ Extracting Unlearned Information from LLMs with Activation Steering")

## 4 Experiments

### 4.1 Models

**Initial Experiments:**
- Utilize WhoIsHarryPotter model [3], which removes Harry Potter-specific knowledge using related sources (books, articles, news posts). Model based on Llama-2-Chat [19] for dialog cases.
- Evaluate alternative unlearning methods by employing base model fine-tuned on the dataset and codebase provided by TOFU authors [7]. Their model based on Phi-1.5.
- For ROME [11], utilize GPT2-XL [14] with authors' code to remove individual facts sequentially.

### 4.2 Datasets

**Harry Potter Unlearning Experiments**
- Dataset curated for experiments: 62 questions using GPT4
- Based on existing dataset by [^16]
- Format: Q&A with "What" or "Who" queries
- Questions chosen to be easily answered by a base model (Llama2 7B)
- Unlearning is reason for incorrect answers, not lack of knowledge in base model

**TOFU Dataset**
- 40 questions provided by authors [^10]
- Focused on two fictitious authors
- Original responses were open-ended
- Transformed into keyword-based Q&A format using key terms from original answers
- Evaluated answer correctness based on extracted keywords

**ROME**
- 20 keyword-based questions from CounterFact dataset [^11] provided by authors
- Questions about historical figure's nationality or mother tongue
- Anonymized keywords replaced using GPT4 to maintain neutrality
- Single question: all possible combinations of anonymized prompts generated.

### 4.3 Experiment Setting

**Experiment Methodology**
- **Following work by**: [^1]
- Apply method to unlearning setting instead of refusal models
- Create input text using:
  - Prompt templates
  - System prompts
  - Questions
  - Answer starts
- First token selected should be the correct keyword
- Compute internal representations (activations) during first token generation
- Calculate mean activations for anonymized prompts
- Subtract mean activations from original text activations to generate **steering vectors**
- Add steering vectors back during sampling of first token only
- Use Temperature value of 2 and Top K value of 40 for diverse answers
- Sample 2000 answers for each question and stop generation at:
  - Harry Potter/ROME: 10 tokens
  - TOFU: 50 tokens

**Ablation Parameters**:
- Vary coefficients adding steering vector during sampling
- Apply method to different layers (multiple layers)
- Use two strategies for generating steering vectors:
  - **Local**: S_l(Q) = A_l(Q) - 1/N ∑n=1^NA_l(Q^*_n)
    - Steering vector for layer l, activation A_l, question Q, anonymized samples Q^*_n
  - **Global**: Take mean of local steering vectors over all questions and use for generation
    - Requires dataset of questions, not easily applicable to real-world settings

**Evaluation Metrics**:
- Calculate frequencies of words (excluding stop words)
- Set probability of answer being correct as maximum frequency value among its words
- Plot **RoC curve** and calculate **AUC score** based on probabilities
  - Ideal model generating only correct answers: AUC score = 1
  - Method pushing correct keywords to be most frequent: also results in AUC score of 1

## 5 Results & Discussion

**Sampling Experiment using Harry Potter Dataset**
- **Initial Sampling**: applied coefficient of 2 and implemented method before final layer of model
- **Local AnonAct Steering** used as it aligns with real-world applications
- Results:
  - Increase in CAFs for many questions, some showing substantial improvements
  - Slight decrease in performance for a small subset of questions
- **Importance of CAF Increases**: goal is to increase CAFs to most common frequency
  - Lowest frequency keyword is still most frequent if no other candidate appears more often

**Evaluation using RoC Plots and "Most Frequent Keywords" Approach**
- Conducted in three settings: Base model, unlearned model, unlearned model with method
- **LLAMA2**: almost perfect score (0.98)
- Unlearned model: score of 0.75, better than random but worse than base LLAMA2
- Method: score of 0.92, closer to base LLAMA2 than unlearned model
- Simple method for determining correct answer yields high AUC score, indicating effective extraction of additional information from the model

**Evaluation on TOFU Model and Dataset**
- CAFs lower or same for almost all questions compared to initial experiments
- Method does not generalize to TOFU unlearning setting

**Experiment with ROME Unlearning Method and CounterFact Dataset**
- Many answers in CounterFact dataset consist of single token
- Probabilities of top tokens plotted for unlearned model without and with method for two example questions
- Method successfully changes prediction, leading sampling away from forced false token
- However, fails to recover original true answer
- Difference between successful Harry Potter case and unsuccessful TOFU/ROME cases:
  - Scope of subject matter:
    - Successful (Harry Potter): large media universe, difficult to retrieve information from any single entity
    - Unsuccessful (TOFU): limited connection between author and birthplace

## 6 Conclusion

**Contribution to Unlearned Model Information Retrieval**
- Present activation steering as a powerful method for this task
- Propose its use in unlearning context, showing effectiveness in recovering lost information from broad subjects (e.g., Harry Potter universe) where concepts are interconnected
- Highlight limitations when applied to narrower settings (granular information deletion)
- Attribute varying performance to differences in how information is unlearned from models
- Mention that broader topics like Harry Potter benefit from activation steering due to the connected in-universe concepts, while narrower topics require a different approach for retrieving unlearned information.

