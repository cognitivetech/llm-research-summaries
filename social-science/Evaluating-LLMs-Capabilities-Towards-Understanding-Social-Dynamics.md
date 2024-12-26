# Evaluating LLMs Capabilities Towards Understanding Social Dynamics

**Author Information:**
- School of Computing and AI, Arizona State University, Tempe AZ (artahir, huanliu@asu.edu)
- University of Illinois Chicago, Chicago IL (lucheng@uic.edu)
- Loyola University Chicago, Chicago IL (msandovalmadrigal, ysilva1@luc.edu)
- School of Social & Behavioral Sciences, Arizona State University, Glendale AZ (d.hall@asu.edu)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
  - [2.1 LLMs in social media analysis](#21-llms-in-social-media-analysis)
  - [2.2 Improving Generative Language models: Fine-tuning vs. Prompt Engineering](#22-improving-generative-language-models-fine-tuning-vs-prompt-engineering)
- [3 Prompt Generation for Detecting and Explaining User behavior](#3-prompt-generation-for-detecting-and-explaining-user-behavior)
  - [3.1 Background](#31-background)
  - [3.2 Prompt Generation](#32-prompt-generation)
  - [3.3 Fine-tuning](#33-fine-tuning)
- [4 Do LLMs understand language in social context?](#4-do-llms-understand-language-in-social-context)
  - [4.1 Similarity Analysis](#41-similarity-analysis)
  - [4.2 Comparison of Social Understanding among LLMs](#42-comparison-of-social-understanding-among-llms)
- [5 Can language models understand directionality in social contexts?](#5-can-language-models-understand-directionality-in-social-contexts)
  - [5.1 Language Models and Directionality Detection](#51-language-models-and-directionality-detection)
  - [5.2 Can LLMs emergent abilities improve directional understanding?](#52-can-llms-emergent-abilities-improve-directional-understanding)
- [6 Can LLM’s detect instances of Cyberbullying and Anti-Bullying?](#6-can-llms-detect-instances-of-cyberbullying-and-anti-bullying)
- [7 Discussion](#7-discussion)
- [8 Conclusion, Limitations, and Future Work](#8-conclusion-limitations-and-future-work)

## Abstract

**Social Media Discourse and Generative Language Models (LLMs)**

**Background:**
- People from diverse backgrounds, beliefs, and motives interact on social media
- Interactions can devolve into toxicity: cyberbullying
- Generative LLMs, like Llama and ChatGPT, have gained popularity for question-answering abilities
- Research question: Can these models understand social media dynamics?

**Key Aspects of Social Dynamics:**
1. **Language**: Understanding language in a social context
2. **Directionality**: Identifying the direction of interactions (bullying or anti-bullying)
3. **Occurrence**: Detecting bullying and anti-bullying messages

**Findings:**
- Fine-tuned LLMs show promising results in understanding directionality
- Mixed results for proper paraphrasing and bullying/anti-bullying detection
- Fine-tuning and prompt engineering improve performance in some tasks

**Conclusion:**
- Understanding the capabilities of LLMs is crucial to develop effective models for social applications.

## 1 Introduction

**Digital Communication Challenges in Social Media Platforms**

**Online Behavior and Cyberbullying**:
- Increased connectivity has brought challenges: toxic online behavior, cyberbullying, harmful content propagation
- Need to address these issues at scale

**Role of Generative Language Models (LLMs)**:
- Recently gained traction for general problem-solving capabilities
- Potential pathway to identify and explain social behaviors
- Could enable promotion of positive interactions

**Research Questions**:
- **RQ1**: Do LLMs understand language in the social context?
- **RQ2**: Which evaluation dimensions expose the weaknesses of LLMs in a social analysis setting?
- **RQ3**: Can LLMs understand directionality in the social context?
- **RQ4**: Can LLMs identify behaviors involved in social dynamics, such as cyberbullying and anti-bullying?

## 2 Related Work

**Prior Research on Social Dynamics** \n This study evaluates the capability of general Language Models (LLMs) in social contexts using **cyberbullying and anti-bullying as case studies**.

### 2.1 LLMs in social media analysis

**LLMs in Social Media Analysis**

**Background:**
- Extensively used in social media analysis (sentiment analysis, emotion detection, irony/sarcasm detection, hate speech detection)
- Transformer based models like BERT and RoBERTa encode words into tokens and learn attention weights for relationship sequences

**Applications:**
1. **Sentiment Analysis and Emotion Detection:**
   - Determine emotional tone in text (sentiment polarity)
   - Utilize models for classification tasks: BERT, GPT-2
   - Emotion detection: joy, anger, sadness
2. **Irony and Sarcasm Detection:**
   - Absence of vocal cues and facial expressions in online discourse
   - LLMs employed to detect instances of irony and sarcasm (RoBERTa)
   - Contextual understanding for literal vs figurative language
3. **Hate Speech and Offensive Language Detection:**
   - Automated tools for detection and mitigation on social media platforms
   - LLMs used to identify hate speech, offensive content
   - Contextual awareness aids in distinguishing between opinions and harmful speech.

### 2.2 Improving Generative Language models: Fine-tuning vs. Prompt Engineering

**Pretraining and Fine-Tuning Language Models (LLMs)**

**Approach**:
- Pretrain a model using abundant textual data
- Fine-tune the model for specific tasks using:
  - New objective function
  - Task-specific dataset

**Rationale**:
- Allows model to learn general-purpose features from pretraining data
- Avoids biases that could result from training on small datasets

**Challenges with Task-Specific Datasets**:
- Difficult to collect, in short supply, or expensive to produce

**Prompt Engineering**:
- Model learns from **prompts** (text templates) instead of task-specific datasets
- Drawback: complexity of identifying correct template structure for each task

**Application of LLMs in Social Media Analysis**:
- Potential to unravel nuanced social dynamics on platforms

## 3 Prompt Generation for Detecting and Explaining User behavior

### 3.1 Background

**Social Media Discourse Analysis**

**Session Structure**:
- Topics organized hierarchically in social media platforms (e.g., subreddits, X hashtags)
- Comments structured as responses to thread topics

**Previous Work**:
- Analyzing useful information from social media sessions (fake news detection, echo chamber formation)

**Problem Setting**:
- Identifying cyberbullying and anti-bullying behavior in social media discourse

**Inputs**:
- Dataset of social media sessions 𝒟={S_1,S_2,...,S_N}
- Each instance d_ij consists of sequence of text tokens from a session S_i

**Outputs**:
- **Behavior Identification (Quantitative)**: Binary classification of behavior as cyberbully or anti-bully
- **Behavior and Motivation Explanations (Qualitative)**: Human-interpretable explanations for identified behaviors

**Enhancing LLM Responses**:
- **Zero-shot prompting**: Providing general prompts to foundational LLMs
- **Chain-of-thought (CoT) prompting**: Constructing prompts that require a rationale before final response
- **Exemplar-based generation**: Providing example generations to guide model responses

### 3.2 Prompt Generation

**Constrained Prompt Generation for Analyzing User Behavior**

**Factors to Consider in Prompt Design:**
- **Task objective**: Identifying and explaining behavior
- **Context**: Topic of conversation subset being analyzed
- **Classification instance**: Differentiating query objective from context
- **Response format**: How the LLM should structure its generated response

**Social Media Conversations:**
- Sequential nature requires tabular format for encoding characteristics
- Instruct-tuning prompt format allows instruction and input to be structured together

**Structured Output Expectation:**
- LLM expected to return output in a manner that allows class labels extraction

**Template Prompt Generation:**
- Templating of procedural generation through constraints

**Figure 1 Example:**
- Sample prompt used for analysis: summarize conversation around post of interest (first question acts as CoT mechanism)

### 3.3 Fine-tuning

**Fine-Tuning Effectiveness Study**
* Prompt: effect of fine-tuning on veracity of responses studied using Low Rank Adaptation (LoRA) strategy [^19]
* LoRA targets specific layers for fine-tuning, reducing parameters to be updated significantly
* Original weight matrix W_0inR^mxn split into AinR^mxr and BinR^rxn
* New weights: W_0+AB
* Backpropagation on A and B allows fine-tuning with fewer parameters

**Model Vulnerabilities:**
* Machine learning models susceptible to mode collapse and catastrophic forgetting when training on specific data types
* Fine-tuning on structural understanding may cause the model to forget previously learned knowledge
* Prevention: combine task-specific data with general instruction data for training. [^36]

**Instruction Data:**
* Used to prevent hazards from fine-tuning
* Alpaca instructions combined with task data at a certain probability.

## 4 Do LLMs understand language in social context?

**Figure 2: Semantic Similarity vs Edit Distance for Paraphrases**
- ChatGPT demonstrates a good balance, while Llama models tend to repeat text verbatim.
- Using exemplars improves performance for the Llama-2 7B model.
- Large language models are predominantly trained on formal and semi-formal text corpora.
- Our objective is to analyze social media discourse, thus we evaluate LLMs' ability to comprehend informal comments.
- Our hypothesis: An LLM understanding a comment can paraphrase it.
- We use comments for two scenarios of paraphrasing: instruction-based and exemplar-based.
  - Instruction-based paraphrasing involves an Alpaca format prompt with instructions asking the model to paraphrase, an input text to be paraphrased, and a response generated by the model.

### 4.1 Similarity Analysis

**Evaluating Language Models (LLMs)**
- **Similarity Metrics**:
    - **BLEU score**: Correspondence between model's output and reference paraphrase based on n-gram overlap
    - **ROGUE score**: Overlap of n-grams with a focus on recall, capturing the presence of reference n-grams in generated text
    - **Jaccard similarity**: Assesses similarity and diversity of sets of n-grams between generated and reference texts
    - **Semantic similarity**: Likeness in meaning between texts, extracted from BERT model encodings
- **Comparison of Similarity Metrics**:
    | Model   | Type     | BLEU score   | ROGUE score  | ROGUE-2 score | ROGUE-L score | Jaccard sim. | Semantic sim. | Levenshtein ratio |
    |---------|----------|--------------|--------------|---------------|--------------|-------------|------------------|--------------------|
    | 13B     | exemplar | 0.725 (0.352) | 0.865 (0.219) | 0.801 (0.299) | 0.859 (0.230) | 0.784 (0.285) | 0.917 (0.124)  | 0.157 (0.228) |
    | 13B     | non_exemplar| 0.687 (0.383)| 0          | 0            | 0            | 0            | 0           | 0.175 (0.260)    |
    | 7B      | exemplar | 0.654 (0.378) | 0.768 (0.352) | 0.734 (0.373) | 0.763 (0.355) | 0.678 (0.354) | 0.888 (0.161)  | 0.239 (0.329) |
    | 7B      | non_exemplar| 0.182 (0.310)| 0          | 0            | 0            | 0            | 0           | 0.733 (0.351)   |
    | CHATGPT | exemplar | 0.187 (0.258) | 0.527 (0.219)| 0.305 (0.264) | 0.493 (0.225) | 0.345 (0.233) | 0.805 (0.127) | 0.521 (0.211) |
    | CHATGPT | non_exemplar| 0.084 (0.094)| 0.456 (0.172)| 0.197 (0.183) | 0.410 (0.165) | 0.264 (0.137) | 0.795 (0.119) | 0.599 (0.145) |
    | GPT2   | exemplar | 0.401 (0.465) | 0.489 (0.427)| 0.426 (0.469) | 0.479 (0.433) | 0.451 (0.433) | 0.788 (0.189) | 0.488 (0.387) |
    | GPT2   | non_exemplar| 0.453 (0.466)| 0.532 (0.439)| 0.480 (0.463) | 0.526 (0.443) | 0.522 (0.418) | 0.800 (0.213) | 0.467 (0.419) |
- **Verbatim vs. High-Quality Generations**:
    - Levenshtein ratio is calculated to differentiate verbatim or close-to-verbatim generations from higher quality generations

### 4.2 Comparison of Social Understanding among LLMs

**LLM Comprehension Analysis**

**Comparing LLMs**: GPT-2, Llama-2 7B/13B, ChatGPT used for analysis.
- Evaluating ability to comprehend social conversations
- Use of Instagram dataset for content analysis

**Common Generation Mistakes**:
1. **Verbatim generation**: Levenshein distance = 0
2. **Repetition of exemplars**
3. **Gibberish generation**: semantically dissimilar text, high edit distance

**Exemplars' Significance**:
- Guides generations towards specific goals
- Only significant change seen in Llama-2 7B variant
- Decrease in Levenshtein distance hints at verbatim repetitions

**Distributional Analysis**:
- Semantic scores favor Llama-2 13B over ChatGPT
- Qualitative analysis and distributional stratification provide context
- Tradeoffs between textual and semantic similarity

**ChatGPT's Understanding**:
- Consistently semantically similar to input
- Avoids being verbatim repetition or gibberish
- Robust understanding of social context

**GPT-2 vs Llama-2 Based Models**:
- Mixture of repetition, valid responses, and gibberish generations
- Possible explanation: Pre-training corpus includes mostly academic text in the case of Llama-2

**Implications**:
- Common metrics like BLEU and ROGUE may not fully capture language understanding
- Multi-faceted evaluation provides a more nuanced picture.

## 5 Can language models understand directionality in social contexts?

**Social Media Mentions and Directionality Patterns in LLMs**
- Social platforms allow users to specify comment targets (mention function)
- This work focuses on LLMs' ability to follow directionality patterns in mentions

### 5.1 Language Models and Directionality Detection

**Phase 1: Adding Structural Understanding**
- First phase of fine-tuning process
- Goal: Divide conditional probability distribution of next word given context of preceding words (P(w\_n|w\_1,w\_2,…,w\_n-1))
- Use large text corpora as datasets for training Language Models (LLMs)
  * Sources include books, articles, websites, etc.

**Phase 1 Challenges:**
- LLMs lack understanding of relationships between comments in social media discourse
- No public data on comment level reasoning in social media conversations
- Use other general purpose structured data as surrogate to learn structure and reasoning (WikiTableQuestions dataset)

**Phase 2: Adding Social Understanding**
- Second phase of fine-tuning process
- Aim: Improve social understanding for under-privileged LLMs
- ChatGPT has better social language understanding compared to competitors
- Study transfer learning abilities of competitor models within problem setting.

### 5.2 Can LLMs emergent abilities improve directional understanding?

**Study Finds Improved Directionality Identification in Language Models with PEFT Fine-Tuning**

**Background:**
- Study evaluates language models' ability to identify directionality using a corpus of 4chan threads and efficient fine-tuning via JORA.
- Objective: determine if PEFT phases enhance model's ability to identify target post for behavior comprehension and discern individual being targeted by the poster.

**Data:**
- Corpus of 4chan threads used as ground truth for directionality information
- Stratified analysis based on number of model parameters and fine-tuning status

**Results:**
- PEFT models significantly outperform base models in target post and reply post identification (Table 2)
- Base models perform better with reply identification due to random mentioning of tags
- PEFT models rely on more educated guesses, resulting in less veracity for reply identification than target identification

**Conclusion:**
- LLMs show promising prospects for learning directionality during fine-tuning.
- Recent findings align with emergent abilities of LLMs.

## 6 Can LLM’s detect instances of Cyberbullying and Anti-Bullying?

**Experiment on Instagram Session Dataset**
* Validating PEFT models' ability to identify cyberbullying and anti-bullying comments
* Uses 100 labeled sessions from Instagram, each containing a post and comments
	+ Allows measurement of model generalization capabilities and addition of context in input prompt
* Stratifies evaluation based on number of model parameters and whether PEFT was applied
* Partitions random sample of comments into cyberbullying and anti-bullying sets
* Ignores comments not falling into either category for binary classification simplicity
* Prompts different variations of LLMs to respond in a constrained manner

**Results**
* All model variations perform close to random chance
* Further investigation reveals models unable to understand social commentary at semantic level
* Non-PEFT models display higher recall (and f1) due to positive labeling of both cyberbullying and anti-bullying cases
* Results consistent with findings in Section 4 regarding LLMs' understanding of informal language on social networks

**Limitations**
* Unable to compare with ChatGPT due to closed source model and safeguards preventing generation of toxic responses.
* Highlights the need for a comprehensive social language comprehension dataset, such as SQuAD, for academic evaluation of popular LLMs.

## 7 Discussion

**Social Data Analysis using Language Models: A Multi-Faceted Problem**

**Dimensions of Social Context Understanding**:
- Generating paraphrases: comparing different models' abilities
- Directionality in social media: understanding social relationships and contexts
- Classification tasks using a combination of directional and social understanding

**Findings**:
- Lack of semantic understanding of language used in social media is the weakest link
- Promising ability of LLMs in directional understanding
- Need for large scale informal training corpuses focused on language comprehension

**Evaluating Language Models' Capabilities Towards Understanding Social Dynamics**:
- Experiments on different dimensions of social context understanding
- Results corroborated in Sections [4](https://arxiv.org/html/2411.13008v1#S4 "4 Do LLMs understand language in social context? ‣ Evaluating LLMs Capabilities Towards Understanding Social Dynamics") and [6](https://arxiv.org/html/2411.13008v1#S6 "6 Can LLM’s detect instances of Cyberbullying and Anti-Bullying? ‣ Evaluating LLMs Capabilities Towards Understanding Social Dynamics")

## 8 Conclusion, Limitations, and Future Work

**LLMs: Recent Popularity and Controversies**
* LLMs (Language Models) have seen **spike in popularity** due to their ability in zero-shot response generation for general-purpose prompts
* Prior studies have questioned [^37] or praised [^20] their abilities in different domains
* Focus: social analysis through the lens of social media discourse, specifically cyberbullying and anti-bullying behavior

**Taxonomy of Learning in Language Models:**
* Prompt engineering
* Fine-tuning
* Constrained generation

**Research Questions:**
* Social context understanding
* Directionality understanding
* Identification of cyberbullying and anti-bullying activity

**Findings:**
* Future breakthroughs require larger coded social datasets
* New model features to better capture social interaction semantics

