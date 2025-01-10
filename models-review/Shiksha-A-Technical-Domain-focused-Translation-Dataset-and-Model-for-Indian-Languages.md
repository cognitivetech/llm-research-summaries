# Shiksha A Technical Domain focused Translation Dataset and Model for Indian Languages

source: https://arxiv.org/html/2412.09025v1
Advait Joglekar 

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Where does Present-day MT fail?](#2-where-does-present-day-mt-fail)
- [3 Related Work](#3-related-work)
- [4 The Dataset](#4-the-dataset)
	- [4.1 First, the Source](#41-first-the-source)
	- [4.2 Data Cleaning and Extraction](#42-data-cleaning-and-extraction)
	- [4.3 Bitext Mining](#43-bitext-mining)
	- [4.4 Data Collation](#44-data-collation)
	- [4.5 Data Analysis](#45-data-analysis)
- [5 The Model](#5-the-model)
	- [5.1 Baseline Model selection](#51-baseline-model-selection)
	- [5.2 Training](#52-training)
	- [5.3 Evaluation](#53-evaluation)
- [6 Translingua](#6-translingua)
- [7 Conclusion](#7-conclusion)
- [8 Limitations](#8-limitations)
- [Appendix A Source Document](#appendix-a-source-document)
- [Appendix B Model Hyperparameters and Results](#appendix-b-model-hyperparameters-and-results)
- [Appendix C Translingua](#appendix-c-translingua)

## Abstract

**Creating Multilingual Parallel Corpora for Neural Machine Translation (NMT)**

**Challenges:**
- Limited exposure to scientific, technical, and educational domains in existing NMT datasets
- Struggles with tasks involving scientific understanding or technical jargon
- Performance is worse for low-resource Indian languages
- Difficulty finding a translation dataset that addresses these specific domains

**Approach:**
- Create a multilingual parallel corpus: English-to-Indic and Indic-to-Indic high-quality translation pairs
- Bitext mining human-translated transcriptions of NPTEL video lectures

**Contributions:**
- Released model and dataset via Hugging Face Co. (<https://huggingface.co/SPRINGLab>)
- Surpass all publicly available models on in-domain tasks for Indian languages
- Improve baseline by over 2 BLEU on average for out-of-domain translation tasks for these Indian languages on the Flores+ benchmark.

## 1 Introduction

**NPTEL (National Programme on Technology Enhanced Learning)**:
- Valuable resource for free, on-demand higher educational content across diverse disciplines
- Curated over 56,000 hours of video lectures, all made publicly available with audio transcriptions
- Supported Indian language transcriptions for over 12,000 hours of video content
- Translations primarily by subject-matter experts

**Benefits**:
- Provides access to university-level educational content in native tongues for a large audience of Indian students
- Helps accelerate the mission of providing accurate Indic subtitles for all NPTEL video lectures

**Translation Pair Counts (in thousands)**:
- **Figure 1**: ![Refer to caption](https://arxiv.org/html/2412.09025v1/x1.png)

**Example Translations from English to Hindi in the Scientific/Technical Domain**:
- **Table 1**: ![Uncaptioned image](https://arxiv.org/html/2412.09025v1/x2.png)
- Sentences marked with † are in-domain, while ‡ are out-of-domain
- Words in blue are terms with multiple meanings that tend to get translated incorrectly
- Words in green represent the correct, expected translation by the model for the blue word in the given context
- Words in red represent incorrect translations.

## 2 Where does Present-day MT fail?

**Performance of Machine Translation Models on Technical-Domain Tasks**
* Google Translate and IndicTrans2 fail to accurately translate "I want to learn the rust language" (Rust programming language, not chemical phenomena) from English to Hindi:
  * Google Translate: "I want to learn the language of war"
  * IndicTrans2: incorrect translation as well
* Importance of understanding context in technical domains for accurate translations:
  * Meaning can change with wrong choice of word
  * Hindi word for "rust" has two meanings (phenomena and war)
* Current models prone to making mistakes in such situations.
* Paper aims to alleviate these shortcomings.

## 3 Related Work

**NPTEL as a Resource for Machine Translation (MT)**
- **Related works**: Samanantar [^10] and IndicTrans2 [^5] identified NPTEL as useful MT resource
- **Data mining**: Mining for parallel sentence pairs using various internet sources, including NPTEL
- **Exact quantity** of mined sentence-pairs from NPTEL not precisely known

**Issues with Data from Related Studies:**
1. **Unfiltered artifacts**: Significant portion of extracted sentence-pairs contained unfiltered artifacts like timestamps
2. **Misidentified code-mixed sentences**: Some code-mixed sentences miscategorized as English, leading to poor alignment quality and underutilization of data
3. **Limited alignments**: Alignments only included 1-1 sentence matchings, leaving room for better alignments with n-m translation pairs
4. **Directionality**: Data solely in English-Indic direction, while significant potential exists for mining Indic-Indic translation pairs from this source

**Addressing Shortcomings in This Work:**
- Aim to improve upon issues found in related studies
- Figure 2 shows average LABSE score across language pairs. (Refer to caption)

## 4 The Dataset

### 4.1 First, the Source

We obtained 10,669 videos' raw data from NPTEL, including bilingual transcriptions in 8 languages (Bengali to Telugu) with English text interspersed. Refer to Appendix A for an example.

### 4.2 Data Cleaning and Extraction

We created a Python script to extract meaningful text from these documents while removing timestamps. The script uses regex patterns and libraries like nltk and indic-nlp to separate English and Indic sentences into parallel document pairs. This creates a massive bitext corpus.

### 4.3 Bitext Mining

We use a well-established approach called Bitext mining to extract accurate sentence pairs from our parallel corpora. Recent work, such as Vecalign, uses multi-lingual embedding models to find similar sentences based on vector similarity. Our method, SentAlign, employs LABSE and optimized alignment algorithms to achieve high accuracy and efficiency, allowing us to also identify 1-n and n-1 sentence matches.

### 4.4 Data Collation

We collected bilingual sentence-pairs from lectures, creating a massive 2.8M pair dataset after removing duplicates.

### 4.5 Data Analysis

**Analysis of Dataset Quality and Quantity**
* Dataset consists of 8 English-Indic and 28 Indic-Indic language pairs
* Common set of lectures among each pair provides inter-Indic alignments for all languages
* 48.6% are English-to-Indic language pairs due to multiple translations
* Primary measure: Average LABSE similarity scores (Figure [2](https://arxiv.org/html/2412.09025v1#S3.F2 "Figure 2 ‣ 3 Related Work ‣ Shiksha: A Technical Domain focused Translation Dataset and Model for Indian Languages"))
* Strong consistency in scores across all languages despite differences in quantity
* Scores tending towards 0.8, never below 0.75
* Validates quality of source data and accuracy of alignments
* Results are in the form of **<bleu>/**<**chrf++**> (Table 2)
* All models evaluated without beam-search or sampling.

## 5 The Model

We aim to evaluate our dataset's value by fine-tuning and testing a powerful MT base model against others. Our hypothesis is that our dataset can improve translation performance in the Technical domain.

### 5.1 Baseline Model selection

We have limited options for a state-of-the-art multi-lingual model that can handle 36 language-pair combinations. We rule out IndicTrans2 as it offers separate models for different directions. NLLB-200 and MADLAD-400 are the remaining candidates, both being Transformer-based models with high potential. We choose NLLB-200 due to its superior performance on Indian languages according to the MADLAD paper's results.

### 5.2 Training

**Model Sizes and Experiments:**
* NLLB-200 models available in various sizes: 600M to 54B parameters
* Choosing 3.3B parameter version for experiments as sweet spot
* Full Fine-Tuning (FFT) approach not feasible due to compute requirements and time
* Parameter-Efficient Fine Tuning (PEFT) method used: Low-Rank Adaptation (LoRA)
* Three approaches for training using LoRA with NLLB 3.3B:
	1. Training on own dataset only, in one direction
	2. Curriculum Learning (CL) using cleaned BPCC corpus and 8 Indian languages (4 million rows), then introducing own dataset
	3. Training on massive 12 million samples including both the cleaned BPCC corpus and own dataset in both directions
* All models trained on a node with 8 NVIDIA A100 40GB GPUs
* Evaluation results for all three models showed similar performance, with third approach performing slightly better.

**Model Training Details:**
* Three different approaches for model training: CL, own dataset only, and massive samples
* PEFT method used: Low-Rank Adaptation (LoRA)
* NLLB 3.3B model used in all experiments
* All models trained on a node with 8 NVIDIA A100 40GB GPUs
* Hyperparameters and detailed results available in Appendix B.

### 5.3 Evaluation

**Model Evaluation:**
* Compare third model (trained on 12 million rows) with baseline NLLB model and IndicTrans2
* In-domain test using top one thousand rows by LABSE score from held-out test set for each language
	+ Model outperforms rest, demonstrating efficacy in technical domain translations
* Tested on Flores+ devtest set:
	+ Able to generalize well as shown by improvements on baseline scores
	+ Close to IndicTrans2 which was trained on larger corpus than ours
* Results depicted in Table [2](https://arxiv.org/html/2412.09025v1#S4.T2) above, language-wise comparison available in Appendix B.

## 6 Translingua

Our models have been integrated into a tool called Translingua, used by human annotators in India to translate NPTEL lectures into multiple languages at high speed and accuracy.

## 7 Conclusion
We introduce Shiksha, a novel Indian language translation dataset and model focused on Scientific, Technical, and Educational domains. With 2.8 million high-quality parallel pairs across 8 languages, our approach improves accuracy and relevance. We fine-tuned state-of-the-art NMT models to achieve significant performance gains in-domain and out-of-domain. Our goal is to highlight the importance of domain-specific datasets for advancing NMT capabilities.

## 8 Limitations

**Limitations of the Translation Dataset and Model:**
* **Heavily skewed towards specific domains**: The dataset is primarily sourced from NPTEL video lectures, focusing on scientific, technical, and educational domains. This may lead to degradation in translation quality for general tasks as standard benchmarks may not catch unexpected ways in which this can affect the model's performance.
* **Lack of diversity**: The dataset covers a narrow range of domains and lacks balance across various contexts. Adding diverse sources, including everyday conversational language, literature, social media, and news articles, is essential to ensure a more stable training process and enhance the system's robustness and accuracy.
* **Limited testing on Indic languages**: The research focused primarily on translating out of English, so the model's performance may not be optimal for Indic-English or Indic-Indic language directions.
* **Dependency on original transcriptions**: The quality of the translation dataset and models is heavily dependent upon the accuracy of the original NPTEL transcriptions. Any errors or inconsistencies in them can affect the training and evaluation process, requiring further human evaluation to ensure the quality of these translations.

## Appendix A Source Document

A bilingual document contains multiple languages in one text.

## Appendix B Model Hyperparameters and Results

Our third approach uses these hyperparameters. Previous approaches were trained for 10 epochs and 4 epochs respectively.

## Appendix C Translingua

[Figure 5: A screenshot from Translingua.](https://arxiv.org/html/2412.09025v1/extracted/6063910/assets/screenshot.png) 

