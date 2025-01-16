# Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains

source: "https://arxiv.org/html/2410.17952v1"

**Study Authors:**
- Ran Xu: Emory University (1) <ran.xu@emory.edu>
- Hui Liu: Amazon (2) liunhu@amazon.com
- Sreyashi Nag: Emory University (2)
- Zhenwei Dai: Emory University (2)
- Yaochen Xie: Emory University (2)
- Xianfeng Tang: Emory University (2)
- Chen Luo: Amazon, Emory University (1) <chen.luo@emory.edu>
- Yang Li: Emory University (2)
- Joyce C. Ho: Emory University (1), Amazon (1) <joyce.c.ho@emory.edu>
- Carl Yang: Emory University (1), Amazon (1) <carlyang@emory.edu>
- Qi He: Emory University (2)

**Affiliations:**
- Emory University: 2 authors
- Amazon: 7 authors, including 3 who are affiliated with both institutions

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Methodology](#3-methodology)
  - [3.1 Problem Setup](#31-problem-setup)
  - [3.2 Stage-I: Retrieval-oriented fine-tuning](#32-stage-i-retrieval-oriented-fine-tuning)
  - [3.3 Stage-II: Domain Adaptive Fine-tuning](#33-stage-ii-domain-adaptive-fine-tuning)
- [4 Experimental Setup](#4-experimental-setup)
  - [4.1 Tasks and Datasets](#41-tasks-and-datasets)
  - [Performance Comparison of LLMs in Science and Computer Science Domains](#performance-comparison-of-llms-in-science-and-computer-science-domains)
  - [4.2 Baselines](#42-baselines)
  - [4.3 Implementation Details](#43-implementation-details)
- [5 Experimental Results](#5-experimental-results)
  - [5.1 Main Results](#51-main-results)
  - [5.2 Ablation Studies](#52-ablation-studies)
  - [5.3 Study on Pseudo-labeled Tuples](#53-study-on-pseudo-labeled-tuples)
  - [5.4 Case Studies](#54-case-studies)
- [6 Conclusion](#6-conclusion)
- [Limitation](#limitation)
- [Appendix A Training Data Details](#appendix-a-training-data-details)
- [Appendix B Test Data Details](#appendix-b-test-data-details)
- [Appendix C Baseline Descriptions](#appendix-c-baseline-descriptions)
- [Appendix D Additional Experimental Results](#appendix-d-additional-experimental-results)
- [Appendix E Prompt Details](#appendix-e-prompt-details)
  - [E.1 Answer Generation](#e1-answer-generation)
  - [E.2 Query Generation](#e2-query-generation)
  - [E.3 Inference](#e3-inference)

## Abstract

**Retrieval-Augmented Generation (RAG) for Specialized Domains:**
* Enhances question answering abilities of large language models (LLMs) by integrating external knowledge
* Challenges in adapting RAG systems to specialized fields like science and medicine: distribution shifts, limited access to domain-specific data

**Proposed Approach: self-training for domain adaptation:**
1. Fine-tune LLM on instruction-following, question-answering, and search-related data
2. Prompt the same LLM to generate diverse domain-relevant questions from unlabeled corpora
3. Filtering strategy to retain high-quality synthetic examples for improvement
4. Leverage these synthetic examples to enhance performance on RAG tasks in specialized domains

**Benefits:**
* Outperforms baselines by 1.2%–8.6% across two backbone sizes and three domains

**Authors:**
* Ran Xu, Hui Liu, Sreyashi Nag, Zhenwei Dai, Yaochen Xie, Xianfeng Tang, Chen Luo, Yang Li, Joyce C. Ho, Carl Yang, Qi He (Emory University and Amazon)

**Contact Information:**
* ran.xu@emory.edu (Ran Xu), liunhu@amazon.com (Liun Hu)

## 1 Introduction

**Retrieval-Augmented Generation (RAG)**

**Enhancing Large Language Models (LLMs)**:
- Technique that enhances LLMs for knowledge-intensive tasks like question answering (QA)
- Incorporates external knowledge sources to customize responses and handle long-tail knowledge
- Avoids the need for costly model retraining
- Reduces "hallucination" of LLMs by ensuring responses are grounded in relevant evidence

**Challenges in Adapting RAG to Specialized Domains**:
- LLMs struggle with distribution shifts and fail to accurately extract information from domain-specific contexts
- Directly using black-box LLMs raises concerns about privacy when dealing with sensitive proprietary data
- Primary challenge is the acquisition of high-quality fine-tuning data for RAG applications

**Proposed Approach: Self-Improving RAG**
- Adapts LLMs' capabilities to generate pseudo-labeled data for domain adaptative QA
- Inspired by the success of self-training in LLM development
- Fine-tunes a single LLM to perform two complementary tasks:
  - **Question answering with context**
  - **Question generation from context**
- Designed to equip LLMs with basic instruction-following and context utilization skills
- Uses unlabeled domain corpora to generate high-quality QA pairs grounded in the context of specialized domains
- Incorporates multiple task types to improve the model's generalization capabilities
- Employs round-trip consistency filtering technique to preserve generated QA pairs

## 2 Related Work

**Retrieval-augmented generation (RAG)**
* Powerful tool in knowledge-intensive NLP tasks: language modeling [^8], question answering [^36]
* Integrate retriever with LLM generator and fine-tune to align capabilities
* Enhancements: dynamical retrieval processes [^27], filter irrelevant contexts [^83], instruction-tuning methods [^44]

**Self-training (pseudo-labeling)**
* One of the earliest approaches to semi-supervised learning [^58]
* Teacher model generates labels for student model fitting
* Widely adopted in NLP tasks: text classification [^16], natural language understanding [^69], ranking [^73]
* Applied to LLM instruction fine-tuning [^86], reasoning [^54], alignment [^21]
* Vulnerable to label noise [^4]
* Sample selection [^37] and reweighting [^74] strategies for stabilization

**Two-stage fine-tuning framework for RAG**
* Fine-tune LLM on retrieval-related data first
* Generate pseudo-labeled tuples by extracting candidate answers from corpus and generating questions conditioned on document and answer
* Filter pseudo-labeled examples with round-trip consistency before further fine-tuning the LLM.

**Domain-specific LLMs**
* Continuous pretraining [^35] or domain-specific fine-tuning [^79] for adaptation to RAG settings
* Few works use strong GPT models for synthetic data generation in RAG scenarios [^89]
* Our method leverages the same LLM for question generation and answering, enabling self-improvement and cost-effective approach.

## 3 Methodology

### 3.1 Problem Setup

**RAG Problem Solution**

**Goal**: Generate answers for queries based on retrieved contexts from a large corpus.
- Retriever (ℛ) used to retrieve top-k relevant documents (𝒟) from the corpus 𝒞
- Language model (LLM, ℳ_θ) generates answer a for query q based on retrieved context 𝒟

**Approach Overview**:
1. **Stage-I**: Learn from retrieval-oriented instruction data in general domain
2. **Stage-II**: Augment 𝒯 with pseudo-labeled tuples (q^', 𝒟^', a^') from specialized domain 𝒞 for self-training
3. **Objective**: Adapt the LLM to specialized domains using 𝒯 ∪ 𝒯^'

**Related Work**:
- Our approach builds upon previous work in Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains (Figure [1](https://arxiv.org/html/2410.17952v1#S2.F1 "Figure 1 ‣ 2 Related Work ‣ ours: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains"))

### 3.2 Stage-I: Retrieval-oriented fine-tuning

**Fine-Tuning Instruction-Finetuned Language Models (LLMs)**
* **Backbone**: Leverage instruction fine-tuned LLMs (e.g., [meta-llama/Meta-Llama-3-8B-Instruct](https://arxiv.org/html/2410.17952v1/meta-llama/Meta-Llama-3-8B-Instruct))
* **Deficiency**: LLMs lack context information for domain-specific questions
* **Improvement**: Fine-tune LLMs with retrieval-oriented tasks

**Fine-Tuning Data:**
1. **General Instruction Fine-tuning (SFT) data**:
   - Maintain models' ability to comprehend and follow instructions
   - Includes OpenAssistant, Dolly, SODA, ELI5, Self-Instruct, and Unnatural Instructions datasets
   - No overlap with target tasks test data
2. **General domain Context-aware QA data**:
   - Bolster LLMs' general RAG skills in generating accurate answers grounded in contexts
   - Includes DROP, NQ, Squad, NarrativeQA, Quoref, ROPES, OpenbookQA, LogiQA, TAT-QA, WebGLM, StrategyQA, BoolQ, FaVIQ, and FEVER datasets
3. **General Retrieval-related Data**:
   - Improve answer generation and query generation skills of LLMs
   - **Answer Generation**: Generate candidate spans from contexts as answers to questions (Squad 1.1 and 2.0 versions, DROP, WebQuestions)
   - **Query Generation**: Generate queries based on documents and answers (NQ, Squad 1.1, StrategyQA, WebQuestions, FaVIQ, FEVER)

**Training Details:**
- Adopt standard instruction finetuning objective
- Compute loss exclusively on tokens of assistant's response.

### 3.3 Stage-II: Domain Adaptive Fine-tuning

**Self-Training Approach for Tailoring LLMs to Specialized Applications**

**Purpose**: To address the distribution shift issue in specialized applications of general domain language models (LLMs)

**Approach**: Self-training using unlabeled corpora from the specialized domain

**Steps**:
1. **Generate pseudo-labeled training samples**:
   - Prompt fine-tuned LLM to generate candidate answers and queries in two steps:
     a. Answer Generation: generate several candidate spans likely to be answers to questions in unlabeled documents
     b. Answer-conditioned Query Generation: generate candidate questions given a candidate answer and document
2. **Strategies**:
   - **Diverse question generation**: create various types of questions (short-span, multiple-choice, claim verification) to improve generalization ability
   - **Data filtering**: retain only high-quality QA pairs where the ground truth answer is present in top-k retrieved documents
3. **Augmentation**: Combine pseudo-labeled samples with SFT data and general domain context-aware QA data for further fine-tuning
4. **Benefits**: Enhance LLMs' question answering abilities within the specialized domain. (See Appendix A for details)

## 4 Experimental Setup

### 4.1 Tasks and Datasets

**Medical Domain Evaluation**
- Evaluated across 11 datasets spanning medical, scientific, and computer science domains
- For medical domain: MIRAGE benchmark (PubMedQA, BioASQ, MedQA, MedMCQA, MMLU medical subsets), LiveQA, and MedicationQA
- Metrics used: accuracy for multiple-choice/True-or-False questions; Rouge-L and MAUVE for open-ended questions; Exact Match (EM) and F1 for Fill-in-the-blank questions.

**Medical Domain Results**
| Datasets      | PubMedQA   | BioASQ    | MedQA     | MedMCQA   | MMLU-med  | LiveQA     | MedicationQA | Avg.        |
|---------------|------------|-----------|-----------|------------|-----------|-------------|--------------|------------|
| Proprietary LLMs, For Reference Only |               |          |           |           |           |             |              |         |
| GPT-3.5       | 67.40      | 90.29     | 66.61     | 58.04      | 75.48     | 42.3 / 62.5 | 36.3 / 46.0   | 62.35       |
| GPT-4         | 70.60      | 92.56     | 82.80     | 66.65      | 87.24     | 44.0 / 65.9 | 41.5 / 59.2   | 69.34       |
| Medical LLMs  |            |          |           |           |           |             |              |         |
| PMC-Llama     | 56.00      | 65.21     | 42.58     | 48.29      | 52.53     | 35.7 / 60.6 | 36.4 / 38.3   | 48.10       |
| MEDITRON      | 56.40      | 76.86     | 49.57     | 52.67      | 65.38     |            |             |          |
| AdaptLLM-v2   | 45.00      | 78.80     | 43.13     | 42.74      | 51.24     | 30.2 / 48.0 | 39.2 / 51.4   | 47.19       |
| BioMistral    | 59.20      | 82.69     | 32.52     | 32.20      | 47.47     | **43.1** / 63.2 | 39.6 / 51.9   | 48.11       |
| MedLlama3    | 74.20      | 83.50     | 61.43     | 61.18      | 77.13     | 27.9 / 45.2 | 29.8 / 35.0   | 59.31       |
| Retrieval-Augmented LLMs                               |            |          |           |           |           |             |              |         |
| Self-RAG      | 71.20      | 73.70     | 48.60     | 44.00      | 53.90     | 35.6 / 54.1 | 39.3 / 46.4   | 52.33       |
| ChatQA1.5     | 66.40      | 82.69     | 42.36     | 46.97      | 61.40     | 39.3 / 65.5 | 39.9 / 48.9   | 54.15       |
| ChatQA1.5 (70B)| 74.80      | 83.17     | 68.89     | 62.54      | 80.51     | 40.1 / 66.3 | **40.8** / 50.2 | 64.40       |
| Backbone: Llama3-8B-Instruct                             |            |          |           |           |           |             |              |         |
| Llama3-8B-it | 64.60      | 88.51     | 55.30     | 58.91      | 69.79     | 34.1 / 54.1 | 37.2 / 45.6   | 58.34       |
| RAFT          | 73.40      | 88.67     | 54.28     | 60.15      | 70.25     | 36.2 / 55.6 | 38.9 / 56.4   | 60.26       |
| EvidenceRAG   | 75.00      | 90.61     | 57.74     | 61.13      | 72.27     | 36.6 / 57.8 | 34.6 / 53.6   | 61.14       |
| ours          | **80.00**  | **91.75** | **62.92** | **67.51**  | **75.57** | **44.4** / 66.6 | **40.1** / 57.4 | **66.04**   |
| w/o Stage II | 78.00      | 90.45     | 60.56     | 58.91      | 69.79     | 42.8 / 62.9 | 38.5 / 55.6   | 64.30       |
| Backbone: Gemma2-27B-Instruct                            |            |          |           |           |           |             |              |         |
| Gemma2-27B-it | 56.20     | 89.32     | 59.70     | 57.30      | 75.67     | 37.4 / 52.8 | 40.2 / 57.0   | 59.40       |
| RAFT         | 67.20      | 91.70     | 62.22     | 61.56      | 78.97     | 39.4 / 62.2 | 40.8 / 56.4   | 63.04       |
| EvidenceRAG  | 63.00      | 90.61     | 62.14     | 61.80      | 79.43     | 34.5 / 58.6 | 34.5 / 44.6   | 60.85       |
| ours         | **73.60**  | **92.07** | **63.63** | **64.16**  | **81.63** | **39.9** / 66.8 | **41.2** / 62.1 | **65.17**   |
| w/o Stage II | 66.00      | 91.59     | 62.45     | 58.67      | 79.61     | 37.2 / 61.6 | 40.8 / 58.6   | 62.33       |

### Performance Comparison of LLMs in Science and Computer Science Domains

**Table 3: Results of our proposed method and baselines in the computer science domain**

**Models**:
- GPT-3.5 (OpenAI)
    - ACC (Multiple Choice): 54.89
    - ACC (Assertion): 67.30
    - Auto (Fill-in-the-blank): 42.93
    - Overall: 55.74
- GPT-4 (OpenAI)
    - ACC (Multiple Choice): 71.48
    - ACC (Assertion): 73.62
    - Auto (Fill-in-the-blank): 56.87
    - Overall: 70.34

**Scientific LLMs**:
- SciTulu 7B (Wadden et al.)
    - ACC (Multiple Choice): 38.40
    - ACC (Assertion): 56.56
    - Auto (Fill-in-the-blank): 27.66
    - Overall: 40.44
- SciTulu 70B (Wadden et al.)
    - ACC (Multiple Choice): 44.24
    - ACC (Assertion): 60.18
    - Auto (Fill-in-the-blank): 31.06
    - Overall: 46.87

**Retrieval-Augmented LLMs**:
- Self-RAG 13B (Asai et al.)
    - ACC (Multiple Choice): 29.87
    - ACC (Assertion): 54.52
    - Auto (Fill-in-the-blank): 30.64
    - Overall: 34.56
- ChatQA 8B (Liu et al.)
    - ACC (Multiple Choice): 35.33
    - ACC (Assertion): 60.18
    - Auto (Fill-in-the-blank): 27.66
    - Overall: 39.11
- ChatQA 70B (Liu et al.)
    - ACC (Multiple Choice): 54.94
    - ACC (Assertion): 62.67
    - Auto (Fill-in-the-blank): 34.89
    - Overall: 38.53

**Backbone: Llama3-8B-Instruct**
- Llama3-8B-it (Meta-AI)
    - ACC (Multiple Choice): 52.69
    - ACC (Assertion): 60.41
    - Auto (Fill-in-the-blank): 26.81
    - Overall: 50.80
- RAFT 8B (Zhang et al.)
    - ACC (Multiple Choice): 54.57
    - ACC (Assertion): 60.86
    - Auto (Fill-in-the-blank): 32.76
    - Overall: 52.38
- EvidenceRAG 8B (Schimanski et al.)
    - ACC (Multiple Choice): 54.42
    - ACC (Assertion): 62.67
    - Auto (Fill-in-the-blank): 35.02
    - Overall: 53.06

**Backbone: Gemma2-27B-Instruct**
- Gemma2-27B-it (Team et al.)
    - ACC (Multiple Choice): 59.96
    - ACC (Assertion): 62.22
    - Auto (Fill-in-the-blank): 40.00
    - Overall: 58.08
- RAFT 27B (Zhang et al.)
    - ACC (Multiple Choice): 60.93
    - ACC (Assertion): 66.06
    - Auto (Fill-in-the-blank): 39.15
    - Overall: 59.07
- EvidenceRAG 27B (Schimanski et al.)
    - ACC (Multiple Choice): 60.63
    - ACC (Assertion): 62.22
    - Auto (Fill-in-the-blank): 40.85
    - Overall: 58.34

**ours**:
- ACC (Multiple Choice): 60.63
- ACC (Assertion): 64.93
- Auto (Fill-in-the-blank): 34.47
- Overall: 57.63
- w/o Stage II: 59.88, 61.99, 34.47, 46.82, 56.55

### 4.2 Baselines

**Categorization of Baselines**

**Groups:**
- **Off-the-shelf general domain LLMs**: GPT-3.5 [^50], GPT-4 [^51], Llama3-8B-it [^46], Gemma2-27B-it [^66]
- **Off-the-shelf domain-specific LLMs**: PMC-llama-13B [^79], MEDITRON-70B [^9], AdaptLLM-v2-8B [^10], BioMistral-7B [^35], MedLlama3-8B [^30], SciTulu 7B and 70B [^70] in both scientific and computer science domains
- **General domain retrieval-augmented LLMs**: Self-RAG-13B [^5], ChatQA1.5-8B and 70B [^44]
- **Domain-specific Retrieval-augmented LLMs**: RAFT [^89], EvidenceRAG [^59] (re-implemented using the same backbones as our approach since they didn't release checkpoints)

**Notes:**
- All baseline models undergo zero-shot evaluation and have context augmented for fair comparison.
- Several domain-specific baselines, such as those which have access to task-specific examples, are not compared.

### 4.3 Implementation Details

**Experimental Setup**

**Models Used**:
- Llama3-it 8B [^46] as base model for Stage II
- Gemma2-it 27B [^66] as base model for Stage I and fine-tuning (LoRA used during fine-tuning due to resource constraints)

**Training Details**:
- Global batch size set to 64 with gradient accumulation of 8
- 1 epoch trained for both stages
- Learning rates: Stage I - 5e-7, Stage II - Llama3 backbone - 2e-7, Gemma backbone - 5e-7
- AdamW optimizer [^45] used with β_1=0.9 and β_2=0.95

**Context Enhancement**:
- Dragon [^39] used to extend context length for models during retrieval

**Retrieval Details**:
- Top-10 retrieval results ensemble from multiple models for evaluation on medical datasets (following MIRAGE benchmark)
- Top-10 passages fetched using Google Search for other datasets

**Hardware**:
- Experiments conducted on 8 NVIDIA A100 GPUs

**Prompt Format**:
- Detailed prompt format provided in Appendix [E](https://arxiv.org/html/2410.17952v1#A5 "Appendix E Prompt Details ‣ ours: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains")

## 5 Experimental Results

### 5.1 Main Results

**Findings from Experimental Results**
- **ours**: consistently outperforms baselines across medical, scientific, and computer science domains and various question-answering formats
  - Average performance gain over Llama variant: 8.01%, 6.37%, 8.61%
  - Average performance gain over Gemma variant: 1.19%, 3.50%, 3.20%
  - Performance comparable to strong proprietary models when using Gemma2-27B as backbone: 93.99%, 98.95%, 86.66% of GPT-4's performance
- **Domain-specific LLMs** (e.g., SciTulu and MedLlama): underperform compared to ours because they are not optimized for retrieval-augmented generation tasks
- **General-domain RAG models** (e.g., ChatQA): face distribution shifts when applied to specialized tasks, struggle to integrate retrieved domain-specific knowledge accurately
- **Domain-specific retrieval-augmented LLMs** (RAFT and EvidenceRAG): show suboptimal performance despite utilizing powerful GPT-4 model for synthetic data generation
  - ours produces more accurate and contextually relevant synthetic question-answer pairs, leading to better downstream performance across all question-answering tasks
- **Computer Science domain**: ours demonstrates promising performance, justifying the potential for adapting our approach to emerging domains.

### 5.2 Ablation Studies

**Retrieval-Oriented Fine-Tuning (Stage I) and Self-Synthesized Data (Stage II)**
* Retrieval-oriented fine-tuning (Stage I): enhances LLM performance on QA tasks compared to original backbone
	+ Significant improvement for Llama3-8B-it: 57.00 → 79.60 in PubMedQA, 81.55 → 91.42 in BioASQ, etc. (Table [1](https://arxiv.org/html/2410.17952v1#S4.T1))
	+ Significant improvement for Gemma2-27B-it: 58.80 → 73.60 in PubMedQA, 89.48 → 90.94 in BioASQ, etc. (Table [2](https://arxiv.org/html/2410.17952v1#S4.T3))
* Self-synthesized data (Stage II): further improvement by generating high-quality synthetic data for target domains
	+ Average increase of 2.21% for Llama and 3.50% for Gemma when fine-tuned on self-synthesized training tuples
* Table [4](https://arxiv.org/html/2410.17952v1#S5.T4) shows consistent performance improvements of ours using Dragon retriever over LLM backbone, demonstrating robustness to different retriever choices.

**Case Studies Comparing Generated Pseudo-Labeled QA Pairs**
* Table 5 presents two case studies comparing generated QA pairs from Stage I fine-tuned model with those generated by the backbone model (Llama3-8B-it).
* In Case Study 1, our model correctly identifies "energy stored in adipocytes can be rapidly released for use at other sites in the body" as supported by provided context after fine-tuning.
* In Case Study 2, our model suggests that cognitive behavioral therapy can help with anxiety after being fine-tuned, which is a correct answer based on context.

### 5.3 Study on Pseudo-labeled Tuples

**Advantages of the Proposed Approach: Question Generation and Filtering**

**Effect of Different Question Generation Models:**
- Comparison of Stage-II using synthetic question-answer pairs generated by Llama-3-8b-it or an off-the-shelf QG model with T5-Large
- Our approach achieves better performance on average, demonstrating the advantage of leveraging fine-tuned models for pseudo-labeled data generation.

**Effect of Question Filtering:**
- Removing low-quality data improves overall model performance and accelerates training process
- Even without filtering, our synthetic questions are highly relevant to the context
- Demonstrates the necessity of incorporating different task types in fine-tuning step for specialized domains.

**Effect of Diverse Question Types:**
- Incorporating various question types (claim verification, multiple choice, short span QA) enhances performance across datasets: PubMedQA, BioASQ, MedQA, MedMCQA, and MMLU
- Removal of short-span QA results in largest performance drops, emphasizing its significance in adapting language models to specialized domains.

### 5.4 Case Studies

**Case Studies Comparing Pseudo-labeled Samples from Baseline Model and Self-Improving Retrieval-Augmented Generation:**

**Case Study 1**:
- Model asked to generate a claim supported by context
- Llama3-8B-it selects sentence from context, making task less challenging for Stage-II training
- Samples produced: simple QA pairs

**Case Study 2**:
- Model tasked with generating answer first, then formulating question based on context and answer
- Llama3-8B-it generates lengthy paraphrased questions that are overly dependent on original text
- Misinterprets context in some cases
- Samples produced: less challenging tasks for Stage-II training due to heavy reliance on context.

**Self-Improving Retrieval-Augmented Generation**:
- After fine-tuning on answer generation and query generation during Stage-I, our model generates higher-quality QA pairs
- These QA pairs are self-contained and understandable without relying heavily on context
- Samples produced: more challenging tasks for Stage-II training due to deeper comprehension of context required.

## 6 Conclusion

**Introduction to Ours: A Framework for Improving LLMs for Domain-Specific Question-Answering Tasks**

Introducing a new instruction fine-tuning framework that enhances Language Models (LLMs) for specialized question-answering tasks by granting them dual capabilities in both question answering and generation. This empowers the creation of diverse, high-quality synthetic questions from unlabeled domain-relevant resources, aiding adaptation to niche fields faced with distribution shifts and limited data.

Through experiments on 11 datasets across three domains, our framework consistently outperforms baseline models, confirming its effectiveness in dealing with the challenges of retrieval-augmented, domain-specific question-answering tasks.

## Limitation

**Limitations of the Approach**

**Single Round Pseudo-Label Generation**:
- Current method relies on a single round of query generation from the corpus
- May restrict the refinement of pseudo label quality
- Iterative refinement of generated synthetic queries could potentially lead to better results

**Additional Training Time**:
- Incorporation of synthetic query generation and filtering adds time complexity
- May affect efficiency in environments with limited computational resources
- However, the method **will not increase the inference time complexity** compared to existing RAG approaches with the same backbone models

**Stronger Query Generation Models**:
- Achieved strong performance with Llama3 8B and Gemma2 27B models
- Leveraging more powerful query generation models, such as Llama-3.1-70B-it, could yield further gains
- Using larger models would incur higher computational costs beyond the current budget.

## Appendix A Training Data Details

We provide the training dataset details (number of examples per stage, instruction format) in Table [6](https://arxiv.org/html/2410.17952v1#A0.T6 "Table 6").

## Appendix B Test Data Details

**Evaluation Datasets:**
- **Medical**:
  - MMLU-med: 6 tasks related to biomedicine with 1089 questions (anatomy, clinical knowledge, professional medicine, human genetics, college medicine, and college biology)
  - MedMCQA: Indian medical entrance exams multiple-choice questions covering 2400 healthcare topics across 21 subjects
  - MedQA: US Medical Licensing Examination multiple-choice questions focused on real-world scenarios from professional medical board exams (1273 questions)
  - BioASQ: 618 biomedical literature questions without ground truth snippets
  - PubMedQA: Biomedical research QA dataset consisting of 1000 manually annotated questions based on PubMed abstracts
  - LiveQA and MedicationQA: Consumer health questions about medications (100 and 674 question-answer pairs, respectively)

**Scientific**:
- **SciQ**: Scientific question-answering dataset containing 13,679 crowdsourced science exam questions in various fields (Physics, Chemistry, Biology)
- **ARC-easy/challenge**: Authentic multiple-choice grade-school level science questions
  - Challenge Set: Questions that stumped both a retrieval-based and word co-occurrence algorithm
  - Easy Set: Less challenging questions
- **MMLU-Sci**: Massive Multitask Language Understanding dataset with 57 tasks across various scientific fields (Physics, Chemistry, Biology)

**Computer Science**:
- **CS-Bench**: Benchmark specifically designed to assess performance of large language models in computer science (around 5,000 test samples, 26 subfields within four major areas)

## Appendix C Baseline Descriptions

**Fine-Tuning Models for Question Answering (QA)**

**Models and their Performance**:
- **Self-RAG**: utilizes instruction fine-tuning to adaptively retrieve passages based on the question and determine if the passage contains useful information for answering the question. [^5]
- **ChatQA**: a fine-tuning pipeline tailored for RAG and conversational QA tasks via aggregating multiple QA and dialogue datasets. [^44]
- **RAFT**: a domain-specific fine-tuning approach that incorporates top-k passages as context during fine-tuning, helping to address discrepancies between training and testing data. [^89]
- **EvidenceRAG**: leverage off-the-shelf LLMs (GPT-4) to generate context-aware question answering datasets, which is then used to fine-tune the student model. [^59]

**Performance Comparison in Scientific Domain**:
| Models | astronomy | college | chemistry | physics | security | geography | macroeconomics | microeconomics | psychology | US history | world history | sexuality | nutrition | virology | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Metrics | ACC | ACC | ACC | ACC | ACC | ACC | ACC | ACC | ACC | ACC | ACC | ACC | ACC | ACC | Avg. |
| GPT-3.5 [(OpenAI, 2022)] | **66.45** | **65.28** | 35.00 | 46.53 | **65.00** | 77.27 | 91.54 | 64.29 | 83.12 | 78.43 | 72.15 | 70.99 | 66.01 | 47.59 | **66.40** |
| GPT-4 [(OpenAI, 2023)] | **93.42** | **93.75** | 61.00 | 73.27 | **91.00** | 94.95 | 97.95 | 94.54 | **96.15** | 95.59 | 94.51 | 93.13 | 89.22 | 56.02 | **87.46** |
| SciTulu 7B [(Wadden et al., 2024)] | 69.74 | 63.89 | 31.00 | 18.63 | 35.00 | **70.20** | 56.58 | 28.43 | 62.00 | 77.43 | 53.06 | 57.38 | 65.65 | 54.90 | 45.78 | **55.95** |
| SciTulu 70B [(Wadden et al., 2024)] | 83.55 | 80.56 | 36.00 | 28.43 | 63.00 | **89.39** | 80.26 | **80.16** | 91.19 | 77.55 | 77.22 | 78.63 | 68.95 | 50.60 | **71.80** |
| Self-RAG 13B [(Asai et al., 2024)] | 55.26 | 58.33 | 24.00 | 21.57 | 35.00 | **60.00** | 61.11 | 21.57 | 67.89 | 58.67 | 58.23 | **53.44** | 43.79 | **40.96** | **48.69** |
| ChatQA 8B [(Liu et al., 2024)] | 60.53 | 54.17 | 29.00 | 33.33 | 35.00 | **70.00** | 64.65 | 33.33 | 74.86 | 49.49 | 54.85 | **59.54** | 57.19 | **45.18** | **54.46** |
| ChatQA 70B [(Liu et al., 2024)] | **82.89** | **79.17** | 46.00 | 48.04 | 36.00 | **84.85** | **80.26** | **84.98** | **91.74** | **86.73** | **82.28** | 74.05 | **77.78** | **51.20** | **75.21** |
| Llama3-8B-it [(Meta-AI, 2024)] | 78.29 | 71.53 | 38.00 | 40.20 | 35.00 | **83.00** | 82.32 | 63.16 | 84.04 | 65.31 | 72.15 | 69.47 | **69.73** | 49.40 | **67.15** |
| RAFT 8B [(Zhang et al., 2024)] | 80.26 | 75.69 | 37.00 | 42.16 | 45.00 | **84.00** | 79.80 | 65.79 | 83.67 | 72.45 | 77.22 | 73.28 | **71.24** | **51.81** | **69.22** |
| EvidenceRAG 8B [(Schimanski et al., 2024)] | 77.63 | 78.47 | 44.00 | 45.10 | 35.00 | **85.00** | 84.85 | 72.37 | 86.24 | 74.49 | 79.32 | 74.05 | **74.84** | **51.20** | **71.59** |
| **ours (8B)** | **85.53** | **81.94** | **47.00** | **50.98** | 45.00 | **88.00** | **89.90** | **76.32** | **84.55** | **92.66** | **83.16** | **81.43** | **84.73** | **54.82** | **77.31** |
| w/o Stage II | 84.87 | 81.25 | **49.00** | 49.02 | 37.00 | 87.00 | 88.89 | 73.68 | 90.64 | 80.61 | 81.01 | 83.21 | **79.41** | 51.81 | 75.95 |
| Gemma2-27B-it [(Team et al., 2024)] | 82.89 | 84.03 | 47.00 | 55.88 | 47.00 | **84.00** | 89.39 | 77.63 | 91.93 | 80.61 | 84.81 | 81.68 | 72.22 | 52.40 | 76.11 |
| RAFT 27B [(Zhang et al., 2024)] | 84.87 | **88.89** | 47.00 | 63.73 | 45.00 | 86.00 | 90.91 | 86.84 | 93.58 | 81.12 | 85.65 | 81.68 | 76.47 | **51.81** | **78.79** |
| EvidenceRAG 27B [(Schimanski et al., 2024)] | 84.87 | 87.50 | 49.00 | 60.78 | 35.00 | **86.00** | **91.41** | 86.84 | 93.94 | 81.63 | 86.08 | 81.68 | 76.80 | **51.81** | **78.84** |
| **ours (27B)** | **90.13** | **91.67** | 49.00 | 68.63 | 45.00 | **87.00** | **92.42** | 85.53 | 87.98 | **95.05** | 83.16 | 81.43 | 84.73 | **55.42** | **81.28** |
| w/o Stage II | 84.21 | 87.50 | 49.00 | 59.80 | 37.00 | 84.00 | 89.90 | 84.21 | 93.58 | 83.16 | 86.50 | 81.68 | 54.82 | **78.38** |

**Backbone Models**:
- Llama3-8B-it (Meta-AI, 2024)
- RAFT (Zhang et al., 2024)
- EvidenceRAG (Schimanski et al., 2024)

## Appendix D Additional Experimental Results

Results of MMLU-sci per task are presented in Table 7 (Appendix C, Baseline Descriptions) - "ours": Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains.

## Appendix E Prompt Details

### E.1 Answer Generation

**[System]**: Generate candidate spans as potential answers to questions in a given passage, distinguishing between entities, verbs, and numbers. Ensure diversity and separate answers using semicolons.

**[Context]**: "Apples are red fruits grown on trees. They can be eaten raw or cooked. Applesauce is a popular recipe made from apples. Cooking apples can also be used to make apple pie."

Candidate Answers: Apples, Red; Eaten raw; Cooked; Applesauce; Apple pie

### E.2 Query Generation

**Question**:
Write a concise question that does not rely on explicit context and is based on the information provided: "What are the guidelines for generating bulleted notes?"

**Answer**:
"Bulleted note creation should follow these guidelines: use multiple headings, emphasize important terms, and review for adherence to the specified format without referencing instructions within the notes."

### E.3 Inference

**Specific Instruction Evaluation Datasets**:
The specific instruction for each evaluation dataset varies based on its question type and can be found in Table 6 ([Link](https://arxiv.org/html/2410.17952v1#A0.T6 "Table 6")).

