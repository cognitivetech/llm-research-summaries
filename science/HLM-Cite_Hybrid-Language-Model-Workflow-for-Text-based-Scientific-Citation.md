# HLM-Cite: Hybrid Language Model Workflow for Text-based Scientific Citation

Qianyue Hao, Jingyang Fan, Fengli Xu 
https://arxiv.org/html/2410.09112

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Problem Formulation](#2-problem-formulation)
- [3 Methods](#3-methods)
  - [3.1 Overview](#31-overview)
  - [3.2 Retrieval Module](#32-retrieval-module)
  - [3.3 LLM Agentic Ranking Module](#33-llm-agentic-ranking-module)
- [4 Experiments](#4-experiments)
  - [4.1 Dataset](#41-dataset)
  - [4.2 Baselines](#42-baselines)
  - [4.3 Overall Performance](#43-overall-performance)
  - [4.4 Ablation Studies](#44-ablation-studies)
  - [4.5 Analysis](#45-analysis)
- [5 Related Works](#5-related-works)
  - [5.1 Pretrained Language Models (PLMs)](#51-pretrained-language-models-plms)
  - [5.2 LLM Agents](#52-llm-agents)
- [6 Conclusions](#6-conclusions)

## Abstract

**Citation Networks in Modern Science:**
* Critical infrastructure for navigation and knowledge production
* Important gap: distinguishing roles of papers' citations
	+ Foundational vs superficial
	+ Logical relationships not always evident from citation networks alone

**Challenges with Large Language Models (LLMs):**
1. New papers may select citations from large existing papers, exceeding LLM context length
2. Implicit logical relationships between papers: surface-level textual similarities may not reveal deeper reasoning required for core citations identification

**Core Citation Concept:**
* Identifies critical references beyond superficial mentions
* Distinguishes core citations from both superficial citations and non-citations

**HLM-Cite: A Hybrid Language Model Workflow for Citation Prediction:**
1. Combines embedding and generative LMs
2. Curriculum finetune procedure to coarsely retrieve high-likelihood core citations from vast candidate sets
3. LLM agentic workflow to rank the retrieved papers through one-shot reasoning, revealing implicit relationships among papers
4. Scales the candidate sets to 100K papers, exceeding size handled by existing methods
5. Demonstrates a 17.6% performance improvement compared to SOTA methods

**Resources:**
* Code open-source at <https://github.com/tsinghua-fib-lab/H-LM> for reproducibility.

## 1 Introduction

**Citation Prediction: Role of Core Citations**

**Background:**
- Modern science research results in increasing annual publications [[1](https://arxiv.org/html/2410.09112v1#bib.bib1)]
- Citation network connects literature and bridges new knowledge with existing [[2](https://arxiv.org/html/2410.09112v1#bib.bib2), [8](https://arxiv.org/html/2410.09112v1#bib.bib8)]
- Goal: Predict which papers in a candidate set will be cited by an emerging new paper (query) [[2](https://arxiv.org/html/2410.09112v1#bib.bib2), [3](https://arxiv.org/html/2410.09112v1#bib.bib3), [4](https://arxiv.org/html/2410.09112v1#bib.bib4), [5](https://arxiv.org/html/2410.09112v1#bib.bib5), [6](https://arxiv.org/html/2410.09112v1#bib.bib6), [7](https://arxiv.org/html/2410.09112v1#bib.bib7)]
- Significance: Reveal hidden information, aid computational social science studies [[9](https://arxiv.org/html/2410.09112v1#bib.bib9), [10](https://arxiv.org/html/2410.09112v1#bib.bib10), [11](https://arxiv.org/html/2410.09112v1#bib.bib11), [12](https://arxiv.org/html/2410.09112v1#bib.bib12), [13](https://arxiv.org/html/2410.09112v1#bib.bib13), [14](https://arxiv.org/html/2410.09112v1#bib.bib14)], save time for researchers [[15](https://arxiv.org/html/2410.09112v1#bib.bib15)]

**Challenges:**
- Existing works treat citation prediction as a binary classification problem, neglecting varying roles of citations [[2](https://arxiv.org/html/2410.09112v1#bib.bib2), [3](https://arxiv.org/html/2410.09112v1#bib.bib3), [4](https://arxiv.org/html/2410.09112v1#bib.bib4), [5](https://arxiv.org/html/2410.09112v1#bib.bib5), [6](https://arxiv.org/html/2410.09112v1#bib.bib6), [7](https://arxiv.org/html/2410.09112v1#bib.bib7)]
- Two major challenges: 
   - Vast candidate sets
   - Implicit logical relationships

**Core Citations:**
- Definition: Core citations depict varying roles of citations [[1](https://arxiv.org/html/2410.09112v1#bib.bib1)]
- Significantly closer relationships with query papers [[1](https://arxiv.org/html/2410.09112v1#bib.bib1)]

**HLM-Cite:**
- Two-stage hybrid language model workflow
   - Stage 1: Retrieve high-likelihood core citations from vast candidate sets using an embedding LM
   - Stage 2: Analyze implicit logical relationships and rank retrieved papers by citation likelihood using a generative LM

**Results:**
- Extensive experiments on cross-field dataset with up to 100K paper candidate sets [[1](https://arxiv.org/html/2410.09112v1#bib.bib1)]
- Performance improvement of 17.6% compared to SOTA baselines [[1](https://arxiv.org/html/2410.09112v1#bib.bib1)]
- Scalable design for handling large candidate sets and complex implicit relationships.

## 2 Problem Formulation

**Paper Citation Relationships**:
- Set of papers denoted as **G**
- Query paper **q** cites a subset of papers
- **Sq** = Papers cited by q
- **Pq** = Papers not cited by q
- **Fq** = Subsequent papers that cite q
- **Core citations** = Most important citations (subset of Sq)
- **Superficial citations** = Less important citations

**Core Citations Characteristics**
- Identified through local citation relationships
- Core citations are likely to be:
  - Critical foundations of the query paper
  - Cited by subsequent papers (Fq)
  - Have more overlapped keywords with query paper
  - More frequently mentioned in main texts
  - Verified using **Microsoft Academic Graph (MAG)** across 19 scientific fields

**Core Citation Prediction Task**
- Given: Query paper **q** and candidate set **Cq**
- **Cq** contains:
  - **tq1** core citations
  - **tq2** superficial citations
  - Possible non-citations
- Goal: Identify **tq1** elements from **Cq** that are core citations

**Text-based prediction approach**
- Uses citation networks only for ground truth
- No network features used beyond textual content
- Learns logical relationships from text
- Applicable to:
  - Previously published papers
  - Ongoing manuscripts without future citations

**Model Characteristics**
- Predicts citations likely to be valued by future papers
- Does not require information about exact future citations
- Uses training and testing sets from previously published papers
- Can be applied to new manuscripts without known subsequent citations

## 3 Methods
### 3.1 Overview
**HLM-Cite Workflow for Text-based Scientific Citation Prediction**
**Components:**
1. **Retrieval Module**:
   - Pretrained text embedding model finetuned with training data
   - Calculates embeddings for query q and each paper in candidate set Cq
   - Based on inner products, retrieves top papers as retrieval set Rq
2. **LLM Agentic Ranking Module:**
   - Analyzes retrieved papers in Rq
   - Collaboratively ranks them according to likelihood of being core citations
3. **Workflow Process:**
   - Given a query q and candidate set Cq, call the retrieval module
   - Calculate embeddings for q and each paper in Cq
   - Retrieve top rq papers from Cq as retrieval set Rq
   - Employ LLM agents in ranking module to analyze and rank the papers in Rq
   - Take top tq1 subscript 1 q end_POSTSUBSCRIPT papers as prediction result.

### 3.2 Retrieval Module

**Model Structure**
- **Text Embedding Model**: GTE-base pretrain model [[16](https://arxiv.org/html/2410.09112v1#bib.bib16)]
  * Based on BERT [[18](https://arxiv.org/html/2410.09112v1#bib.bib18)]
  * 768-dimensional dense vectors for text embedding
  * Multi-stage contrastive learning tasks
  * 110M parameters, frozen lower layers
- **Finetuning**: Higher 5 layers only

**Curriculum Finetuning**
- Adapting general-corpus model to specific task
- Two-stage curriculum: easy to hard

**Stage 1:**
- Distinguish core citations from non-citations
- Exclude interference of superficial citations (hard negatives)
- Construct training data with one query, core citation, and non-citations
- Cross-entropy loss for classification error

**Stage 2:**
- Consider ranking task: core citations, superficial citations, and non-citations
- NeuralNDCG loss function [[20](https://arxiv.org/html/2410.09112v1#bib.bib20)] to measure difference between model output and ground truth ranking
- One query with multiple core citations, superficial citations, and non-citations in training data
- In-batch negative sampling [[21](https://arxiv.org/html/2410.09112v1#bib.bib21)] for reducing embedding cost.

### 3.3 LLM Agentic Ranking Module

**Overall Procedure**
- Incorporate LLMs' reasoning capability to rectify ranking of retrieved papers by core citation likelihood
- Three agents: analyzer, decider, and guider, driven by LLMs and collaborate via natural language communications
- Given a query paper and possible core citations, analyze the logical relationship between each individual paper in the retrieval set and the query paper using the analyzer
- Obtain revised ranking of likelihoods of becoming core citations from the decider based on analysis provided by the analyzer
- Guider produces a one-shot example under human supervision to enhance complex reasoning and assist the analyzer and decider via the chain of thought (CoT) method

**Analyzer: from Textual Similarity to Logical Relationship**
- Predicting citations requires in-depth understanding of logical relationships among papers, not just textual similarity
- Analyzer extracts why the query paper cites each candidate by leveraging the implicit knowledge base encoded in the LLM [[23](https://arxiv.org/html/2410.09112v1#bib.bib23), [24](https://arxiv.org/html/2410.09112v1#bib.bib24), [25](https://arxiv.org/html/2410.09112v1#bib.bib25)]

**Decider: Final Ranking for Core Citation Prediction**
- Generates final ranking of core-citation likelihoods based on analysis of paper relationships obtained from the analyzer
- Provides explanations alongside rankings to improve rationality of results [[26](https://arxiv.org/html/2410.09112v1#bib.bib26), [27](https://arxiv.org/html/2410.09112v1#bib.bib27)]

**Guider: One-Shot Learning**
- Selects a representative query paper and several candidates outside the test set
- Produces exemplary analysis and rectified ranking through the analyze-decide procedure [[22](https://arxiv.org/html/2410.09112v1#bib.bib22)]
- Manually reviews and revises obtained texts to ensure correct revelation of logical relationships between query paper and candidates
- Concatenates revised texts at the beginning of prompts for the analyzer and decider via the chain of thought (CoT) method [[22](https://arxiv.org/html/2410.09112v1#bib.bib22)]

## 4 Experiments

### 4.1 Dataset

**Experimental Dataset**

**Microsoft Academic Graph (MAG)**:
- Archives hundreds of millions of research papers across 19 major scientific domains
- Forms a huge citation network

**Dataset Filtering**:
- Traverse the dataset and filter 12M papers with:
  - Abundant **core citations**
  - Superficial **citations**
- Randomly sample 450,000 queries and subsequently sample 5 core citations and 5 superficial citations for each query
- Randomly divide the sampled queries into 8:2 as training and testing sets

**Scientific Domains**:
- **Natural Science**: Biology, Chemistry, Computer Science, Engineering, Environmental Science, Geography, Geology, Materials Science, Mathematics, Medicine, Physics
- **Social Science**: Art, Business, Economics, History, Philosophy, Political Science, Psychology, Sociology

**Dataset Statistics**:

| **Scientific Domain** | **Training Set** | **Testing Set** | **Total** | **Query Count** | **Candidate Count**
|---|---|---|---|---|---|
| Natural science       | 386,655         | 3,830,273      | 483,880   | 479,596        | 474,4912
| Social science        | 13,345          | 169,727       | 161,200    | 20,404        | 205,088
| **Total**            | 400,000         | 4,000,000     | 500,000   | 500,000       | 4,950,000

**Notes**:
- The dataset is filtered to include only papers with abundant core citations and superficial citations
- The sampled queries are divided into training and testing sets
- Natural science queries may cite papers from the social science domain, and vice versa.

### 4.2 Baselines

**Baselines for Evaluation:**
- **Simple rule-based method**: Predict core citations based on keyword overlap between candidate and query papers.
- **Baselines based on Language Models (LMs) for scientific texts**:
  - SciBERT [[31](https://arxiv.org/html/2410.09112v1#bib.bib31)]: Pretrained on millions of research papers from Semantic Scholar using same approaches as BERT.
  - METAG [[6](https://arxiv.org/html/2410.09112v1#bib.bib6)]: Learns to generate multiple embeddings for various kinds of patterns in citation network relationships.
  - PATTON and SciPATTON [[5](https://arxiv.org/html/2410.09112v1#bib.bib5)]: Finetuned with network masked language modeling and masked node prediction tasks on citation networks from BERT and SciBERT respectively.
  - SPECTER [[3](https://arxiv.org/html/2410.09112v1#bib.bib3), [32](https://arxiv.org/html/2410.09112v1#bib.bib32)]: Continuously pretrained with contrastive objective from SciBERT.
  - SciNCL [[4](https://arxiv.org/html/2410.09112v1#bib.bib4)]: Improvement of SPECTER by considering hard-to-learn negatives and positives in contrastive learning.
  - SciMult [[33](https://arxiv.org/html/2410.09112v1#bib.bib33)]: Multi-task contrastive learning framework focusing on finetuning models with common knowledge sharing across different scientific literature understanding tasks.
- **Baselines based on Universal Embedding Models**:
  - BERT [[18](https://arxiv.org/html/2410.09112v1#bib.bib18)]: Pretrained with masked language modeling and next sentence prediction objectives on Wikipedia and BookCorpus.
  - GTE [[16](https://arxiv.org/html/2410.09112v1#bib.bib16), [34](https://arxiv.org/html/2410.09112v1#bib.bib34)]: Series of top embedding models finetuned from BERT with multi-stage contrastive learning task.
  - OpenAI-embedding-ada-002 and OpenAI-embedding-3 [[New](https://openai.com/index/new-embedding-models-and-api-updates/)]: Advanced universal embedding models proposed by OpenAI.

### 4.3 Overall Performance

**Curriculum Finetuning**:
- Conducted using batch sizes of 512 and 96
- Trained for 10 epochs
- Took approximately 12 hours on 8×NVIDIA A100 80G GPUs

**OpenAI API Access**:
- Used to access GPT models for LLM agentic ranking
- Kept using GPT-4 as the guider, but alternated two versions of GPTs for the analyzer and decider

**Performance Results**:
- **Table 2**: Overall performance comparison of various models in Natural Science, Social Science domains
- **Metric:** PREC@3/5, NDCG@3/5
- **Models:** SciBERT, METAG, PATTON, SciPATTON, SPECTER, SciNCL, SciMult-vanilla, SciMult-MoE, SPECTER-2.0, BERT-base, BERT-large, OpenAI-ada-002, OpenAI-3, GTE-base, GTE-base-v1.5, GTE-large, GTE-large-v1.5, H-LM (GPT3.5), H-LM (GPT4o)
- **Findings**: Significant improvement over baselines in all scientific domains and metrics; overall PREC@5 improvement up to 17.6%

**Evaluation**:
- Set vast candidate sets with tq=10K for all models, retrieval size rq=8
- Evaluated performance via PREC@3/5 and NDCG@3/5

**Case Study**:
- Representative testing sample: DNA Nanorobot paper [[35](https://arxiv.org/html/2410.09112v1#bib.bib35)]
- Analyzer correctly reveals core citations informing the key design or query paper [[36](https://arxiv.org/html/2410.09112v1#bib.bib36), [37](https://arxiv.org/html/2410.09112v1#bib.bib37)]
- Analyzer correctly identifies superficial citations inspiring some design details [[38](https://arxiv.org/html/2410.09112v1#bib.bib38)]
- Decider ranks retrieval set based on rational analysis, improving precision

### 4.4 Ablation Studies

**Validation of Designs through Ablation Studies:**
* Conduct studies to verify curriculum finetuning of retrieval module and LLM agents design in ranking module
* Results shown in Table [3](https://arxiv.org/html/2410.09112v1#S4.T3 "Table 3 ‣ 4.4 Ablation Studies ‣ 4 Experiments ‣ HLM-Cite: Hybrid Language Model Workflow for Text-based Scientific Citation Prediction")

**Curriculum Design Validation:**
* Delete first and second stages of curriculum to calculate metrics on retrieval set
* Performance drop indicates enabling adaption of pretrained model from easy to hard, improving transfer performance

**Ablation Studies:**
* Remove analyzer or guider to test impact on performance
* Absence of agents leads to degradation, proving essential role in process
* Statistical significance verified in Appendix [A.5.2](https://arxiv.org/html/2410.09112v1#A1.SS5.SSS2 "A.5.2 Ablation Studies ‣ A.5 Statistical Significance ‣ Appendix A Appendix ‣ HLM-Cite: Hybrid Language Model Workflow for Text-based Scientific Citation Prediction")

**Table 3:**
* Bold indicates best performance
* Model | Natural science | Social science | Overall
  ---|---|---|---
  **Full curriculum**: 0.683, 0.598, 0.705, 0.641, 0.704, 0.623, 0.724, 0.663, 0.684, 0.598, 0.706, 0.641
  **w/o Stage1**: 0.682, 0.595, 0.703, 0.638, 0.706, 0.623, 0.724, 0.662, 0.682, 0.596, 0.704, 0.639
  **w/o Stage2**: 0.666, 0.587, 0.686, 0.626, 0.685, 0.614, 0.705, 0.650, 0.667, 0.588, 0.687, 0.627
* **Full workflow**: 0.725, 0.644, 0.734, 0.677, 0.743, 0.661, 0.751, 0.693, 0.725, 0.644, 0.735, 0.677
* **w/o Analyzer**: 0.723, 0.629, 0.733, 0.666, 0.736, 0.648, 0.747, 0.684, 0.723, 0.630, 0.734, 0.667
* **w/o Guider**: 0.686, 0.594, 0.707, 0.638, 0.702, 0.618, 0.723, 0.660, 0.686, 0.595, 0.708, 0.639
* **w/o Analyzer&Guider**: 0.659, 0.580, 0.688, 0.626, 0.686, 0.608, 0.712, 0.651, 0.660, 0.581, 0.689, 0.627

### 4.5 Analysis

**HLM-Cite Workflow Analysis**

**Advantage of Large-Scale Candidate Sets:**
- Regardless of candidate size, method significantly surpasses baselines (Figure 4)
- Larger sets lead to higher relative performance improvement up to 18.5% in tq=100K

**Effect of Retrieval Size:**
- rq subscript 𝑟𝑞r_q is a key hyperparameter balancing work between modules
- Increasing rq subscript 𝑟𝑞r_q leads to higher recall rates and potential for better ranking
- Too many retrieved candidates confuse analysis, leading to low-quality ranking and performance drop
- Optimal value is supposed to be around 8 for natural and social science (Appendix A.4)

**Effect of One-shot Example:**
- CoT enhances LLMs by demonstrating logical reasoning structure rather than specific knowledge
- No significant difference between one-shot and few-shot learning (Table 4)

**Effect of LLM Types:**
- Substituting GPT-3.5 with other open-source LLMs shows performance differences
- Larger LLMs perform slightly better, but lightweight models perform significantly worse than GPT models (Table 5)
- Importance of implicit knowledge in large-scale parameters for solving tasks like citation prediction.

## 5 Related Works

### 5.1 Pretrained Language Models (PLMs)

**Pretrained Language Models:**
* **Masked token prediction models**: trained via different objectives [[18](https://arxiv.org/html/2410.09112v1#bib.bib18), [41](https://arxiv.org/html/2410.09112v1#bib.bib41)]
* **Contrastive learning models**: [[3](https://arxiv.org/html/2410.09112v1#bib.bib3), [42](https://arxiv.org/html/2410.09112v1#bib.bib42), [4](https://arxiv.org/html/2410.09112v1#bib.bib4), [16](https://arxiv.org/html/2410.09112v1#bib.bib16)]
* **Permutation language modeling**: [[43](https://arxiv.org/html/2410.09112v1#bib.bib43)]
* Advantages: fewer computational resources, suitable for various tasks on large-scale corpus (classification, clustering, retrieval) [[17](https://arxiv.org/html/2410.09112v1#bib.bib17)]

**Generative Large Language Models (LLMs):**
* Pretrained on vast corpus
* Exhibit strong few-shot and zero-shot learning ability [[23](https://arxiv.org/html/2410.09112v1#bib.bib23), [24](https://arxiv.org/html/2410.09112v1#bib.bib24)]
* Superior performance on text analyzing, code generation and solving math problems [[44](https://arxiv.org/html/2410.09112v1#bib.bib44), [45](https://arxiv.org/html/2410.09112v1#bib.bib45), [46](https://arxiv.org/html/2410.09112v1#bib.bib46), [47](https://arxiv.org/html/2410.09112v1#bib.bib47), [48](https://arxiv.org/html/2410.09112v1#bib.bib48), [49](https://arxiv.org/html/2410.09112v1#bib.bib49), [50](https://arxiv.org/html/2410.09112v1#bib.bib50)]
* Lack of combination with small embedding models

**Hybrid Language Workflow:**
* Incorporates small embedding models' advantage (efficient large-scale retrieval) and generative LLMs' capability (textual reasoning)

### 5.2 LLM Agents

**Applications of Language Models (LLMs)**

**Decision Making**:
- LLM agents successful in sandbox games [[51](https://arxiv.org/html/2410.09112v1#bib.bib51), [52](https://arxiv.org/html/2410.09112v1#bib.bib52)]
- Robot controlling [[53](https://arxiv.org/html/2410.09112v1#bib.bib53)]
- Navigation [[54](https://arxiv.org/html/2410.09112v1#bib.bib54)]

**Simulating Social Life**:
- LLM agents simulate daily social life [[55](https://arxiv.org/html/2410.09112v1#bib.bib55)]
- Generate physical mobility behavior [[56](https://arxiv.org/html/2410.09112v1#bib.bib56)]
- Reveal macroeconomic mechanisms [[57](https://arxiv.org/html/2410.09112v1#bib.bib57)]

**Natural Language Processing**:
- Role-fused LLM agents collaboratively solve tasks [[45](https://arxiv.org/html/2410.09112v1#bib.bib45), [27](https://arxiv.org/html/2410.09112v1#bib.bib27), [58](https://arxiv.org/html/2410.09112v1#bib.bib58)]

**Limitations**:
- Difficulty handling tasks with extremely long texts, such as citation precision on vast candidate sets

**Approach in this Paper**:
- Incorporate generative LLMs with embedding models to enable working on very large candidate sets.

## 6 Conclusions

**Task Investigation: Scientific Citation Prediction**
- Introduces core citation concept to enhance conventional task by distinguishing key citations
- Proposes hybrid language model workflow combining embedding and generative capabilities for improved performance
- Demonstrates superiority in tasks with large candidate sets through experiments and analysis
- Identifies limitation of LLMs: illusion problem, where unfaithful analysis may occur under certain circumstances and specific samples may be affected
- Highlights need for verifying LLM output and improving hybrid workflow reliability as future research directions.
