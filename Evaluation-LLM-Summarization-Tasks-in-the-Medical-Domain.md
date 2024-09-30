# Evaluation of Large Language Models for Summarization Tasks in the Medical Domain: A Narrative Review

Baohua Zhang, Yongyi Huang, Wenyao Cui, Huaping Zhang
https://arxiv.org/abs/2409.18170

## Contents
- [1 Abstract](#1-abstract)
- [2 Introduction](#2-introduction)
- [3 Human Evaluations in Electronic Health Record Documentation](#3-human-evaluations-in-electronic-health-record-documentation)
  - [3.1 Criteria for Human Evaluations](#31-criteria-for-human-evaluations)
  - [3.2 Analysis of Human Evaluations](#32-analysis-of-human-evaluations)
  - [3.3 Drawbacks of Human Evaluations](#33-drawbacks-of-human-evaluations)
- [4 Pre-LLM Automated Evaluations](#4-pre-llm-automated-evaluations)
  - [4.1 Categories of Automated Evaluation](#41-categories-of-automated-evaluation)
  - [4.2 Drawbacks of Automated Metrics](#42-drawbacks-of-automated-metrics)
- [5 FUTURE DIRECTIONS](#5-future-directions)
  - [5.1 Zero-Shot and In-Context Learning](#51-zero-shot-and-in-context-learning)
  - [5.2 Parameter Efficient Fine-Tuning](#52-parameter-efficient-fine-tuning)
  - [5.3 Parameter Efficient Fine-Tuning with Human-Aware](#53-parameter-efficient-fine-tuning-with-human-aware)
  - [5.4 Drawbacks of LLMs as Evaluators](#54-drawbacks-of-llms-as-evaluators)
- [6 Evaluation Needs for the Clinical Domain](#6-evaluation-needs-for-the-clinical-domain)

## 1 Abstract
**Narrative Review on Large Language Models for Summarization Tasks in Medical Domain:**
- **Background:**
  - Large Language Models have advanced clinical Natural Language Generation (NLG)
  - Managing medical text volume in high-stakes industry
- **Challenges:**
  - Evaluation of these models remains a significant challenge
- **Current State of Evaluation:**
  - Overview of current methods used for evaluation
- **Proposed Future Directions:**
  - Addressing resource constraints for expert human evaluation.

## 2 Introduction
* LLMs' development has led to advancements in NLG field
* Significant potential for medical domain, especially reduction of cognitive burden through summation tasks like question answering
* Challenges: ensuring reliable evaluation of performance and addressing complexities of medical texts and LLM-specific challenges (relevancy, hallucinations, omissions, factual accuracy)
* Healthcare data can contain conflicting or incorrect information
* Current metrics insufficient for nuanced needs of medical domain and unable to differentiate between various users' needs
* Automation bias adds potential risks in clinical settings
* Efficient automated evaluation methods necessary.

**Background:**
* LLMs showing promise in reducing cognitive burden in medical domain
* Recent advancements allow processing extensive textual data for summarizing entire patient histories
* Challenge: reliable evaluation of performance, especially with complex medical texts and LLM-specific challenges.

**Limitations of Current Metrics:**
* Simple extractive summarization metrics perform adequately but fall short in abstractive summarization tasks requiring complex reasoning and deep medical knowledge
* Unable to account for relevancy needs of various users.

**Medical Domain Complexities:**
* Conflicting or incorrect information in healthcare data complicates LLM challenges
* Consequences of inaccuracies can be severe due to automation bias.

**Future Directions:**
* Overcome labor-intensive human evaluation process through automated methods.

## 3 Human Evaluations in Electronic Health Record Documentation

**Human Evaluations in Electronic Health Record Documentation**

**Pre-GenAI Rubrics for Clinical Notes Evaluation**:
- Based on pre-GenAI rubrics that assess clinical documentation quality
- Variability based on type of evaluators, content, and analysis required
- Flexibility allows for tailored evaluation methods, capturing task-specific aspects

**Expert Evaluators**:
- Crucial role in maintaining high standards of assessment
- Field-specific knowledge allows for accurate evaluation

**Commonly Used Pre-GenAI Rubrics**:
- **SaferDx [6]**: Identifies diagnostic errors, analyzes missed opportunities
- **Physician Documentation Quality Instrument (PDQI-9) [7]**: Evaluates physician note quality across 9 criteria
- **Revised-IDEA [8]**: Offers feedback on clinical reasoning documentation

**Criteria Emphasized in Pre-GenAI Rubrics**:
- Omission of relevant diagnoses throughout the differential diagnosis process
- Relevant objective data, processes, and conclusions associated with those diagnoses
- Correctness of information, free from incorrect, inappropriate, or incomplete data
- Additional questions based on specific clinical documentation usage

**Evaluation Styles**:
- **Revised-IDEA**: Count style assessment for 3 of 4 items to ensure minimum inclusion
- **SaferDx**: Retrospective analysis of GenAI use in clinical practice

**Adapting Pre-GenAI Rubrics for LLM-Generated Content**:
- New and modified rubrics address unique challenges posed by LLM-generated content
- Emphasize safety [14], modality [15, 16], and correctness [17, 18]

### 3.1 Criteria for Human Evaluations

**Criteria for Human Evaluations of LLM Output (Large Language Model)**

**1. Hallucination**:
- Captures unsupported claims, non-sensical statements, improbable scenarios, and incorrect or contradictory facts in generated text
- Examples: Unfounded medical claims, nonsensical statements, implausible scenarios, factual errors, inconsistencies

**2. Omission**:
- Identifies missing information in a generated text
- Medical facts, important information, critical diagnostic decisions can be considered omitted if not included
- Examples: Overlooking key details, neglecting essential facts, leaving out crucial steps or considerations

**3. Revision**:
- Questions about making revisions to generate text
- Ensures generated texts meet specific standards set by researchers, hospitals, or government bodies

**4. Faithfulness/Confidence**:
- Grades whether a generated text preserves source content and reflects confidence and specificity present in the source text
- Evaluates if generated text maintains coherence with original material and presents accurate conclusions

**5. Bias/Harm**:
- Examines potential harm to patients or bias in responses of generated texts
- Questions about inaccurate, irrelevant, poorly applied information that could negatively impact patients

**6. Groundedness**:
- Assesses quality of source material evidence for a generated text
- Evaluates reading comprehension, recall, reasoning steps, and adherence to scientific consensus

**7. Fluency**:
- Grades coherence, readability, grammatical correctness, and lexical correctness of a generated text
- Ensures that the text flows well and is easy to understand.

### 3.2 Analysis of Human Evaluations

**Binary Categorizations:**
- Breakdown complex evaluations into simpler decisions
- True/False or Yes/No response schema
- Penalizes smaller errors by making responses acceptable or unacceptable

**Likert Scales:**
- Higher level of specificity in the score
- Ordinal scale with as many levels as necessary
- Introduces more problems meeting assumptions of a normal distribution
- Complex and can lead to disagreement among reviewers

**Count/Proportion Based Evaluations:**
- Identify pre-specified instances of correct or incorrect key phrases
- Precision, recall, f-score or rate computed from evaluator's annotations
- Numerical score for a generated text based on these metrics

**Edit Distance Evaluations:**
- Annotate errors in the generated text and make edits until satisfactory
- Corrections of factual errors, omissions, irrelevant items
- Evaluative score is the distance from original to edited version based on characters, words, etc.
- Levenshtein distance algorithm used to calculate this distance

**Penalty/Reward Schemas:**
- Assign points for positive outcomes and penalize negative ones
- Similar to national exam schemas with weighted trade-offs between false positives and false negatives
- Provides a high level of specificity in assigning weights representative of the trade-off between false positives and false negatives.

### 3.3 Drawbacks of Human Evaluations 

**Resource-intensive**:
- Nuanced assessments but
- Reliant on clinical domain knowledge recruitment

**Evaluator influence**:
- Experience and background impact interpretations
- Evaluative instructions shape assessment personal interpretations and beliefs

**Limited resources**:
- Number of evaluators limited by time and finances
- Manual effort requires clear guidelines for inter-rater agreement

**Training required**:
- Human evaluators need training to align with rubric's intent
- Time constraints limit availability of medical professionals

**Evaluation framework validity concerns**:
- Lack of details about framework creation
- Insufficient reporting of inter-rater reliability

**Evaluation rubrics limitations**:
- Not specifically designed for LLM-generated summaries assessment
- Focus on human-authored note quality evaluation elements only

## 4 Pre-LLM Automated Evaluations

**Pre-LLM Methods for Text Quality Assessment**

**Advantages of Automated Metrics:**
- Practical solution to resource constraints
- Used extensively in fields like NLP (Question answering, translation, summarization)
- More efficient in terms of time and labor

**Dependence on Reference Texts:**
- Effectiveness closely tied to quality and relevance of gold standards
- Heavy reliance on high-quality reference texts for accurate evaluations

**Challenges:**
- Struggle to capture nuance, contextual understanding in complex domains (clinical diagnosis)
- Implications of subtle differences in phrasing or reasoning are significant.

### 4.1 Categories of Automated Evaluation

**Automated Evaluation Categories**
- **Word/Character-based**: Relies on comparisons between a reference text and generated text to compute an evaluative score. Can be based on character, word, or sub-sequence overlaps. Examples: ROUGE (N, L, W, S), Edit distance metrics
- **Embedding-based**: Creates contextualized or static embeddings for comparison instead of relying on exact matches between words/characters. Captures semantic similarities between texts. Example: BERTScore
- **Learned metric-based**: Trains a model to compute evaluations, either on example scores or reference and generated text pairs. Examples: Crosslingual Optimized Metric for Evaluation of Translation (COMET)
- **Probability-based**: Calculates likelihood of a generated text based on domain knowledge, references, or source material. Penalizes off-topic information. Example: BARTScore
- **Pre-Defined Knowledge Base**: Relies on established databases of domain-specific knowledge to inform evaluations. Valuable in specialized fields like healthcare. Examples: SapBERTScore, CUI F-Score, UMLS Scorer

### 4.2 Drawbacks of Automated Metrics

**Drawbacks of Automated Metrics for LLMs**
- Prior to advent of Language Models (LLMs), automated metrics generated single score representing quality of a text regardless of length or complexity
- Single score approach can make it difficult to pinpoint specific issues in the text and understand contributing factors
- In case of LLMs, nearly impossible to understand precise factors contributing to a particular score
- Automated metrics offer speed but rely on surface-level heuristics such as lexicographic and structural measures
- These fail to capture more abstract summarization challenges like clinical reasoning and knowledge application in medical texts

## 5 FUTURE DIRECTIONS
**Complementing Human Expert Evaluators**:
- LLMs can serve as evaluators to complement human expert evaluators

**Prompt Engineering Stages**:
1. **Zero-Shot and In-Context Learning (ICL)**: Fitting an LLM into a larger schema for training and prompting it to evaluate other LLMs
2. **Parameter Efficient Fine Tuning (PEFT)**: Enhancing the ability of an LLM to align its outputs with human preferences through instruction tuning and reinforcement learning with human feedback (RLHF)
3. **PEFT with Human Aware Loss Function (HALO)**: Further improving the accuracy and performance of LLMs for evaluative tasks

**Advantages of LLM-Based Evaluations**:
- **Speed and consistency**: Provide advantages similar to traditional automated metrics
- **Direct engagement with content**: Offer more information into factual accuracy, hallucinations, and omissions compared to simplistic heuristics used in human evaluations
- **Scalability**: Address the limitations of manual assessment in complex domains

**Early Studies on LLM-Based Evaluations**:
- Demonstrated their utility as an alternative to human evaluations
- Hold promise for addressing the shortcomings of traditional automated metrics and human evaluations.

### 5.1 Zero-Shot and In-Context Learning
**Prompting Strategies:**
- **Zero-Shot**: Model given task description without examples before generating output.
- **Few-Shot (In-Context Learning)**: Provides task description with a few examples to guide responses.
  - Number of examples varies based on model's architecture and optimal performance point.
  - Typically, between one and five examples are used.

**Hard Prompting:**
- Enables LLMs to perform tasks not explicitly trained for.
- Performance can vary depending on pre-training relevance.

**Anatomy of an Evaluator Prompt:**
1. **Prompt**: Task description and instructions.
2. **Information**: Necessary data for making evaluations.
3. **Evaluation**: Guidelines and formatting of the evaluation.

**Soft Prompting (Machine-Learned):**
- Adds learnable parameters as virtual tokens to a model's input layer.
- Fine-tunes the model's behavior without altering core weights.
- Outperforms few-shot prompting in large-scale models.
- May be necessary for optimal task execution when prompting alone does not suffice.

### 5.2 Parameter Efficient Fine-Tuning 
**Challenges for LLMs**:
- Struggle with tasks requiring domain-specific knowledge or handling nuanced inputs
- Supervised fine-tuning (SFT) methods with Parameter Efficient Fine-Tuning (PEFT) can be employed to address these challenges

**Parameter Efficient Fine-Tuning (PEFT)**:
- Involves training on a specialized dataset of prompt/response pairs tailored to the task at hand
- **Quantization**: Reduces time and memory costs by using lower precision data types (4-bit, 8-bit) for LLMs weights
- **Low rank adaptors (LoRA)**: Freeze the weights of a LLM and decompose them into a smaller number of trainable parameters

**Benefits of PEFT**:
- Refines an LLM by embedding task-specific knowledge
- Ensures the model can respond accurately in specialized contexts
- Performance improvements are directly tied to the quality and relevance of prompt/response pairs used for fine-tuning
- Narrows focus of the LLM to task-specific behaviors, such as medical diagnosis or legal reasoning

### 5.3 Parameter Efficient Fine-Tuning with Human-Aware

**Human Alignment Fine-Tuning with Human-Aware Loss Function**

**Purpose**: Align LLM with human values and preferences during fine-tuning

**Methods for Human Alignment Training:**
1. **Reinforcement Learning with Human Feedback (RLHF)**: Updates LLM to produce higher-scoring responses using a reward model and Proximal Policy Optimization (PPO)
2. **Direct Preference Optimization (DPO)**: Streamlines training by optimizing model outputs directly based on human preferences, without the need for an explicit reward model

**Comparison of Methods:**
- PPO improves LLM performance but is sample-inefficient and can suffer from reward hacking
- DPO is more sample-efficient and better aligned with human values as it focuses on desired outcomes

**Recent Developments**:
1. **Direct Policy Optimization (DPO) Variants**: Joint Preference Optimization (JPO), Simple Preference Optimization (SimPO), Kahneman-Tversky Optimiza-tion (KTO), and Pluralistic Alignment Framework (PAL) have emerged to improve alignment training methods, prevent over-fitting, and address heterogeneous human preferences.
2. Regularization terms and modifications to the loss function are introduced in alternative methods to ensure robust alignment.
3. Alternative modeling assumptions used in these methods can prevent breakdown of DPO's alignment when direct preference data is not available.

**Application in Medical Field**: Training data from human evaluation rubrics on a smaller scale can be incorporated into a loss function designed for human alignment using DPO.

### 5.4 Drawbacks of LLMs as Evaluators
- **Rapid pace of evolution**: outpaces ability to thoroughly validate before use in practice
- **Lack of sufficient mathematical justification**: for new optimization techniques
- Difficulty in allocating time and resources for proper validation, compromising reliability
- Sensitivity to prompts and inputs: highly variable output based on internal knowledge representation and pre-training schema
- **Egocentric bias**: could affect evaluations as more LLM generated text appears in source texts

**Challenges in using LLMs as evaluators:**
- Stringent testing and safety checks required to mitigate risks
- Ensuring fairness, particularly in sensitive domains like healthcare
- Continuous evaluation, testing, and refinement needed for reliability and safety.

**Human Aware Loss Functions (HALOs): Development Timeline**
- First introduced with Proximal Policy Optimization (PPO) in 2017
- Since then, several HALO algorithms have been developed: 
  - Rejection Sampling
  - IPO: Identity Preference Optimization
  - cDPO: Conservative DPO
  - KTO: Kahneman Tversky Optimization
  - JPO: Joint Preference Optimization
  - ORPO: Odds Ratio Preference Optimization
  - rDPO: Robust DPO
  - BCO: Binary Classifier Optimization
  - DNO: Direct Nash Optimization
  - TR-DPO: Trust Region DPO
  - CPO: Contrastive Preference Optimization
  - SPPO: Self-Play Preference Optimization
  - PAL: Pluralistic Alignment Framework
  - EXO: Efficient Exact Optimization
  - AOT: Alignment via Optimal Transport
  - RPO: Iterative Reasoning Preference Optimization
  - NCA: Noise Contrastive Alignment
  - RTO: Reinforced Token Optimization
  - and SimPO: Simple Preference Optimization.

## 6 Evaluation Needs for the Clinical Domain

**Clinical Domain Evaluation Needs**
- **Reliable evaluation strategies** important for GenAI validation as healthcare focuses on clinical safety
- Human evaluations: high reliability but time-consuming
- **Automated evaluations**: promising alternative to human evaluations but have limitations in the clinical domain
  * Traditional non-LLM automated evaluations overlook hallucinations, assess reasoning quality poorly, and struggle with text relevance
- LLMs as potential alternatives for human evaluators
  * Must consider unique requirements of the clinical domain
  - **Well-designed LLM evaluator**: could combine high reliability of human evaluations with efficiency of automated methods
  - Offer best of both worlds: ensure clinical safety without sacrificing assessment quality.

