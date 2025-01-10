# A COMPREHENSIVE EVALUATION OF LARGE LANGUAGE MODELS ON MENTAL ILLNESS

by Abdelrahman Hanafi, Mohammed Saad, Noureldin Zahran, Radwa J. Hanafy, and Mohammed E. Fouda
https://arxiv.org/pdf/2409.15687

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
  - [1.1 Evaluating LLMs on Mental Health Tasks](#11-evaluating-llms-on-mental-health-tasks)
  - [1.2 Fine-tuning LLMs for mental health tasks](#12-fine-tuning-llms-for-mental-health-tasks)
  - [1.3 LLMs for Data Augmentation and Chatbot Development](#13-llms-for-data-augmentation-and-chatbot-development)
  - [1.4 Benchmarks for Evaluating LLMs in Psychiatry](#14-benchmarks-for-evaluating-llms-in-psychiatry)
  - [1.5 Literature Reviews](#15-literature-reviews)
  - [1.6 Research Gaps](#16-research-gaps)
- [2 Methodology Implementation](#2-methodology-implementation)
  - [2.1 Experimental Setup\*\*](#21-experimental-setup)
  - [2.2 Datasets](#22-datasets)
  - [2.3 Sampling](#23-sampling)
  - [2.4 Models](#24-models)
  - [2.5 Prompt Templates](#25-prompt-templates)
  - [2.6 Evaluation Metrics](#26-evaluation-metrics)
  - [2.7 Parsing Model Outputs](#27-parsing-model-outputs)
  - [2.8 Additional Considerations](#28-additional-considerations)
- [3 Results \& Discussion](#3-results--discussion)
  - [3.2 Performance Variability Experiment](#32-performance-variability-experiment)
  - [3.3 Binary Disorder Classification (Task 1)](#33-binary-disorder-classification-task-1)
  - [3.4 Disorder Severity Evaluation (task 2)](#34-disorder-severity-evaluation-task-2)
  - [3.5 Psychiatric Knowledge Assessment (task 3)](#35-psychiatric-knowledge-assessment-task-3)
- [4 Additional Investigations](#4-additional-investigations)
- [5 Challenges, Conclusions and Future Directions](#5-challenges-conclusions-and-future-directions)

## Abstract
**Study Findings on Large Language Models (LLMs) in Mental Health Tasks**

**Background:**
- LLMs have shown promise in various domains, including healthcare
- This study evaluated LLMs on mental health tasks using social media data

**Participants:**
- Various LLMs: GPT-4, Llama 3, Claude, Gemma, Gemini, Phi-3, etc. (models ranging from 2 billion to 405+ billion parameters)

**Tasks and Methods:**
- Binary disorder detection
- Disorder severity evaluation
- Psychiatric knowledge assessment
- Utilized 9 main prompt templates across tasks

**Key Findings:**
1. **Binary Disorder Detection:**
   - Models like GPT-4 and Llama 3 exhibited superior performance, reaching accuracies up to 85% on certain datasets
   - Prompt engineering played a crucial role in enhancing model performance
2. **Disorder Severity Evaluation:**
   - Few-shot learning significantly improved the model’s accuracy, highlighting the importance of contextual examples
   - Phi-3-mini model showed a substantial increase in performance with over 6.80% improvement in balanced accuracy (BA) and nearly 1.3 drop in mean average error (MAE) when transitioning from zero-shot to few-shot learning
3. **Psychiatric Knowledge Assessment:**
   - Recent models generally outperformed older, larger counterparts
   - Llama 3.1 405b achieved an accuracy of 91.2%

**Challenges:**
- Variability in performance across datasets
- Need for careful prompt engineering
- High cost associated with using large models
- Limitations imposed by the quality of social media data
- Ethical guards imposed by LLM providers hinder accurate evaluation due to non-responsiveness to sensitive queries.

## 1 Introduction

**Artificial Intelligence in Mental Healthcare: An Introduction**

**Background:**
- Rapid transformation of mental healthcare landscape by AI
- Global burden of mental illness: 5.1% of global disease burden, affecting 280 million people worldwide and resulting in US$14 trillion economic costs [1][2]
- Integration of AI in psychiatry offers solutions for early detection, diagnosis, prediction of treatment outcomes, and therapeutic interventions

**Applications:**
- Analysis of various data modalities: neuroimaging, physiological signals, genetics, demographics [3]
- Natural Language Processing (NLP) applications: analyzing text from clinical interviews and self-reports [4][6]
  * Early NLP applications focused on linguistic patterns in text using Linguistic Inquiry and Word Count [4]
  * Significant turning point with the emergence of transformer models like BERT [5]

**Large Language Models (LLMs):**
- Substantial improvements in tasks like sentiment analysis, suicide risk assessment, detection of mental health conditions from social media posts [6]
- Trained on massive amounts of text data from diverse sources across the internet, encompassing billions of words and wide range of human knowledge
- Potential applications: initial consultations, analyzing patient sessions, administrative tasks, virtual counselors, monitoring social media for signs of mental distress [11]
  * Enhancing efficiency, accessibility, and quality of care provided to patients [11]

**Challenges:**
- Proper verification for real medical context needed before widespread implementation
- Current research focuses on evaluation, fine-tuning, data augmentation and chatbots, benchmarks, literature reviews.

**Evaluation (Sec. 1.1):**
- Ensuring effectiveness and safety in practical applications

**Fine-Tuning (Sec. 1.2):**
- Adapting LLMs to specific mental health tasks and contexts

**Data Augmentation and Chatbots (Sec. 1.3):**
- Generating synthetic data for training models
- Developing chatbots to improve patient engagement and accessibility

**Benchmarks (Sec. 1.4):**
- Establishing standardized evaluation metrics for mental health applications of AI

**Literature Reviews (Sec. 1.5):**
- Examining the current state of research in using AI for mental healthcare and identifying future directions.

### 1.1 Evaluating LLMs on Mental Health Tasks

**Study Evaluating Language Models (LLMs) on Mental Health Tasks:**
* Earliest work: study by [12] on GPT-3.5-Turbo's performance for stress, depression, and suicidality detection using social media posts
	+ F1 scores: 73% (stress), 86% (depression), 37% (suicidality) were unsatisfactory, especially in severe predictions
* Extension of evaluation by [13]: broader range of affective computing tasks
	+ Big five personality prediction and sentiment analysis assessment of GPT-3.5-Turbo
	+ Decent results but did not surpass fine-tuned RoBERTa model
	+ Manual API usage might have affected results.

### 1.2 Fine-tuning LLMs for mental health tasks

**Fine-tuning Language Models for Mental Health Tasks**

**MentaLlama Model**:
- Fine-tuned on IMHI dataset: social media posts labeled with mental health conditions and explanations
- Dataset created by combining multiple existing mental health datasets and generating additional labels using GPT-3.5-Turbo
- MentaLlama-chat-13B model outperforms or approaches state-of-the-art discriminative methods in 7 out of 10 test sets
- Generates explanations on par with GPT-3.5-Turbo, providing interpretability

**Evaluation of Multiple LLMs**:
- [15] evaluated Alpaca, Alpaca-LoRA, FLAN-T5, Llama 2, GPT-3.5-Turbo, and GPT-4 on various mental health prediction tasks using online text data
- Explored ZS prompting, FS prompting, and instruction fine-tuning techniques
- Fine-tuned Alpaca and FLAN-T5 on six mental health prediction tasks across four Reddit datasets
- Results show that instruction fine-tuning significantly improves LLM performance on these tasks
- **Mental-Alpaca** and **Mental-FLAN-T5** outperform larger models like GPT-3.5-Turbo and GPT-4 on specific tasks
- Perform on par with the state-of-the-art task-specific model Mental-RoBERTa [6]
- Includes an exploratory case study on models' reasoning capability, suggesting both future potential and limitations of LLMs.

### 1.3 LLMs for Data Augmentation and Chatbot Development

**LLMs for Data Augmentation and Chatbot Development**

**Chat-Diagnose System [16]**:
- Explainable and interactive LLM-augmented system for depression detection in social media
- Incorporates a tweet selector, image descriptor, and professional diagnostic criteria (DSM-5) to guide the diagnostic process
- Uses chain-of-thought (CoT) technique to provide explanations and diagnostic evidence
- Leverages answer heuristics from a traditional depression detection model in full-training setting
- Achieves SoTA performance in various settings, including zero-shot (ZS), few-shot (FS), and full-training scenarios

**Generating Synthetic Reddit Posts [17]**:
- Investigates using GPT-3.5-Turbo for generating synthetic Reddit posts simulating depression symptoms from the Beck Depression Inventory (BDI)-II questionnaire
- Aims to enhance semantic search capabilities
- Finds that using original BDI-II responses as queries is more effective, suggesting the generated data might be too specific for this task

**Developing Chatbots for Clinical Diagnosis [18]**:
- Develops chatbots simulating psychiatrists and patients in clinical diagnosis scenarios, specifically for depressive disorders
- Follows a three-phase design: collaboration with psychiatrists, experimental studies, and evaluations with real psychiatrists and patients
- Emphasizes an iterative refinement of prompt design and evaluation metrics based on feedback from both psychiatrists and patients
- Findings reveal differences in questioning strategies and empathy behaviors between real and simulated psychiatrists

### 1.4 Benchmarks for Evaluating LLMs in Psychiatry

**Benchmarks for Evaluating LLMs in Psychiatry (PsyEval)**

**Approach Differences**:
- In contrast to aforementioned studies, PsyEval aimed to create a comprehensive benchmark for assessing LLM capabilities in mental health domain.
- Instead of evaluating existing LLMs on established datasets, they designed specific tasks to test LLM abilities.

**Components of PsyEval Benchmark**:
- **Mental Health Question-Answering**: Evaluates LLMs' ability to provide accurate and informative responses related to mental health conditions, symptoms, and treatments.
- **Diagnosis Prediction**: Tasks focused on predicting diagnoses using data from social media or simulated dialogues between patients and clinicians.
- **Empathetic and Safe Psychological Counseling**: Assesses LLMs' ability to provide empathetic and safe psychological counseling in simulated conversations.

### 1.5 Literature Reviews

**Literature Reviews on Language Models (LLMs) in Mental Health Care**

**Contribution Papers:**
- Comprehensive overview of opportunities and risks associated with LLMs in psychiatry [11]
- Systematic review of applications of LLMs in psychiatry [20]
- Scoping review focusing on applications, outcomes, and challenges of LLMs in mental health care [21]

**Opportunities:**
- Enhance diagnostic accuracy
- Personalized care
- Streamlined administrative processes
- Efficient analysis of patient data
- Summarize therapy sessions
- Complex diagnostic problem-solving
- Automate managerial decisions (e.g., personnel schedules, equipment needs)

**Risks:**
- Labor substitution
- Reduced human-to-human socialization
- Amplification of existing biases

**Systematic Review [20]:**
- Identified 16 studies on LLMs in psychiatry applications (clinical reasoning, social media analysis, education)
- Found promise for LLMs assisting with tasks like diagnosing mental health issues and managing depression
- Highlighted limitations: difficulties with complex cases, potential underestimation of suicide risks

**Scoping Review [21]:**
- Identified 34 publications on LLMs in mental health care (new datasets, model development, employment evaluation)
- Most studies utilized GPT-3.5 or GPT-4 with some fine-tuning
- Datasets sourced from social media platforms and online mental health forums
- Challenges: data availability and reliability, handling mental states, appropriate evaluation methods.

### 1.6 Research Gaps

**LLMs in Mental Health Care: Research Gaps**

**Research Gaps**:
- Studies employ outdated LLM models like GPT-3.5-Turbo, GPT-4, and Llama 2
- No investigations into newer models such as GPT-4-0, GPT-4-Turbo, Llama 3, Phi-3, Gemma 1 & 2, Claude models, and others
- Lack of focus on **prompt engineering** to investigate effects on disorder detection tasks
- Overlooked factors: LLM variation across runs, effect of small prompt modifications, lack of transparency in reporting methodologies

**Goal of the Work**:
- Comprehensively evaluate capabilities of LLMs on mental illness symptom/mental illness detection from social media

**Contributions**:
- Test models never used before in this context: Gemini 1.0 Pro, GPT-4/GPT-4-Turbo/GPT-4o/GPT-4o mini, Mistral NeMo/Medium/Large/Large 2, Claude 3/3.5, Mixtral, Gemma 1/2, Phi-3, Llama 3/3.1, and Qwen 2
- Conduct investigation of published human-annotated mental health datasets
- Extensive experiments with various prompting methods: FS prompting, severity prompting
- Unique experiments: temperature variation tests, filtering strategies, chain-of-thought prompting
- Provide comprehensive details about methodology and prompts for transparency and reproducibility
- List drawbacks, challenges, and suggestions for future researchers

## 2 Methodology Implementation

### 2.1 Experimental Setup**
- Designed series of experiments to evaluate LLMs' capabilities in mental health tasks
- Three primary tasks: binary disorder detection, disorder severity evaluation, psychiatric knowledge assessment

**Task 1: ZS Binary Disorder Detection**
- ZS learning paradigm used for evaluating LLM understanding of psychiatric disorders
- Tasked with determining a user's mental disorder based on social media post and task description (depression, suicide risk, stress)

**Task 2: ZS/FS Disorder Severity Evaluation**
- FS learning technique employed to help models better understand context and complexities of the task
- Evaluated LLMs on depression and suicide risk severity assessment using both ZS and FS approaches
- Comparison of results between these two methods determined how additional context improved model accuracy
- FS prompting only used for severity evaluation due to its complexity and better demonstration of potential improvement through FS learning.

**Task 3: ZS Psychiatric Knowledge Assessment**
- Tested LLM's knowledge of basic psychiatric concepts by presenting multiple choice questions related to psychiatry.

### 2.2 Datasets

**Datasets**
- Avoided automated labeling methods due to unreliability
- Prioritized datasets labeled by experts or crowdsourced workers trained by experts
- Focused on depression, suicide risk, and stress due to prevalence and availability of high-quality datasets
- Other disorders not included due to data collection challenges and limitations in publicly available datasets

**Datasets Summary**
- **Dreaddit**: Stress Classification, Reddit Crowdsourced (1,000 posts) [Publicly Available]
- **DepSeverity**: Depression Severity Classification, Reddit Professionals (3,553 posts) [Publicly Available]
- **Stress-Annotated Dataset (SAD)**: Stress Severity Classification, Daily Stressors (9 categories), SMS-like conversations and LiveJournal data [Crowdsourced 6,850 sentences] [Publicly Available]
- **DEPTWEET**: Depression Severity Classification, X (formerly Twitter) Crowdworkers [40,191 tweets] [Available Upon Request]
- **CSSRS-Suicide**: Suicide Risk Severity Assessment, Reddit (r/SuicideWatch) Professional psychiatrists (500 Reddit users) [Publicly Available]
- **SDCNL**: Suicide vs. Depression Classification, Reddit (r/SuicideWatch, r/Depression) N/A (1,895 posts) [Publicly Available]
- **PsyQA**: Generating Long Counseling Text, Various Mental Health Disorders, Medical entrance exams [22K questions, 56K answers] [Available Upon Request]
- **MedMCQA (Psychiatry Subset)**: Medical Question Answering, Various psychiatric disorders [4,421 (~5%) publicly available]
- **MedSym**: Symptom Identification, 7 mental disorders (Depression, Anxiety, ADHD, Disorder, OCD, Eating Disorder), Reddit [8,554 sentences] [Available Upon Request]
- **PsyQA (Psychiatry Subset)**: Medical Question Answering, Various clinical topics [47,457 questions] [Publicly Available]
- **LGBTeen**: Emotional Support and Information, Queer-related topics, Reddit (r/LGBTeens) [1,000 posts, 11,320 LLM responses] [Publicly Available]
- Other datasets [Cascalheira et al., Hua et al., Mukherjee et al., DATD, PAN12, PJZC, RED SAM, CLPsych15]

**Additional Tasks**
- Emotional Support and Counseling: not included in this study but relevant datasets exist for evaluation.

**Datasets for Stress Analysis and Depression Detection in Social Media Posts**

1. **Dreaddit**
   - Collected from Reddit using PRAW API
   - Contains 187,444 posts from 10 subreddits across five domains: interpersonal conflict (abuse, social), mental illness (anxiety, PTSD), and financial need (financial)
   - Divided into 5-sentence segments; crowdworkers labeled as "Stress," "Not Stress," or "Can't Tell"
   - Final dataset includes 3,553 labeled segments indicating stress expression and negative attitude
2. **DepSeverity**
   - Designed for depression severity identification in social media posts
   - Contains 3,553 Reddit posts labeled across four levels: minimal, mild, moderate, severe
   - Created by relabeling an existing binary-class dataset using the Depressive Disorder Annotation scheme
   - Annotated by professional annotators with majority voting for disagreements
3. **SDCNL (Suicidal Ideation vs. Depression in Social Media Posts)**
   - Addresses lack of research on distinguishing suicidal ideation from depression
   - Contains 1,895 posts from r/SuicideWatch and r/Depression subreddits
   - Employed unsupervised approach to correct initial labels based on subreddit origin
4. **SAD (Stress Annotated Dataset)**
   - Designed for stressor identification in text to facilitate appropriate interventions
   - Contains 6,850 sentences labeled across nine categories of stressors derived from various sources
   - Crowdworkers annotated stressor categories and severity ratings on a 10-point Likert scale
5. **DEPTWEET (Depression Severity in Twitter)**
   - Contains 40,191 tweets labeled across four depression severity levels: non-depressed, mild, moderate, severe
   - Created by collecting tweets based on depression-related keywords and annotating them using a typology based on DSM-5 and PHQ-9 clinical assessment tools
   - Annotation process involved 111 crowdworkers and majority voting for final labels
6. **MedMCQA (Medical Multiple Choice Questions)**
   - Large-scale dataset for medical domain question answering with 194k MCQs covering various healthcare topics
   - Intended to serve as a benchmark for evaluating AI models in medical question answering
7. **RED SAM (Reddit Social Media Depression Levels)**
   - Designed for depression severity detection in social media posts
   - Contains 16,632 Reddit posts labeled across three categories: "Not Depressed," "Moderately Depressed," and "Severely Depressed"
   - Annotated by two domain experts following specific guidelines to determine depression level
8. **CSSRS (Columbia Suicide Severity Rating Scale) - Suicide (Excluded)**
   - Designed for assessing suicide risk severity in individuals based on social media posts
   - Contains 500 Reddit users from r/SuicideWatch subreddit labeled across five levels: Supportive, Indicator, Ideation, Behavior, and Attempt
   - However, the dataset was discarded due to challenges evaluating fairness and costs.

### 2.3 Sampling

**Cost-Saving Strategies for Testing Large Models**

**Strategies Employed**:
- Use pre-split test dataset if available
- Sample 1000 instances from datasets exceeding 1000 examples

**Rationale for Sampling**:
- Sample size of 1000 chosen based on **Hoeffding's inequality (Equation 1)**
- Provides a general upper bound for error difference between sample and population
- With N=1000 and ε=0.05, the probability of deviation exceeding 5% is only 0.0135
- This implies a greater than **98.5% probability** that error will remain within 5% range of true error

**Binarization of Datasets**:
- Binarized datasets with:
  - "False" label (0) for minimum severity score
  - "True" label (1) for any higher severity scores

### 2.4 Models

**Model Selection for Experiments**
- Wide range of models selected: large, state-of-the-art and smaller, resource-constrained
- Included both closed-source (APIs) and open-source models
- Models from various families: OpenAI's GPT series, Anthropic's Claude models, Mistral AI’s models, Google's models, Meta's Llama series, Phi-2, and MentaLlama.
- Attempted to include Gemini 1.5 Pro but faced constraints (rate limits and challenges)

**Summary of Models Used in Evaluation:**
| Model Size (Parameters) | Source                    | API Service Provider       |
|---|---|---|
| Claude 3 Haiku [52]      | Proprietary            | Anthropic                |
| Claude 3 Sonnet [52]     | Proprietary            | Anthropic                |
| Claude 3 Opus [52]       | Proprietary            | Anthropic                |
| Gemma 2b [54]             | 2B                      | Google                   |
| Gemma 7b [54]             | 7B                      | Google                   |
| Gemma 2 9b [55]          | 9B                      | Google                   |
| Gemma 2 27b [55]         | 27B                     | Google                   |
| Phi-3-mini [8]           | Proprietary            | None                     |
| Phi-3-small [54]          | Proprietary            | None                     |
| Phi-3-medium [14B]        | Proprietary            | None                     |
| Qwen 2 72b [72B]         | 72B                     | Alibaba Cloud           |
| Llama 3.1 [70B]          | Proprietary            | Meta                    |
| Llama 3.1 [405B]         | Proprietary            | Meta                    |
| Mistral 7b [7B]           | Proprietary            | Mistral AI              |
| Mixtral 8x7b [45B]       | 45B                     | Mistral AI              |
| Mixtral 8x22b [141B]      | 39B                     | Mistral AI              |
| Together AI Mistral NeMo | Proprietary            | None                     |
| Phi-2 [2.7b]             | Proprietary            | Phi-1 Labs              |
| GPT-4o mini [56]         | Proprietary            | OpenAI                  |
| GPT-3.5-Turbo [175B]      | Proprietary            | OpenAI                  |
| GPT-4o [175B]            | Proprietary            | OpenAI                  |
| GPT-4 [7]               | Proprietary            | OpenAI                  |
| Llama 3 [8b]             | 8B                      | Meta                    |
| Llama 3.1 [8b]           | 8B                      | Meta                    |
| Together AI Llama 2 [13B] | Proprietary            | None                     |
| Together AI Llama 2 [70B] | Proprietary            | None                     |
| Mistral Large [Proprietary] | Mistral AI              | Proprietary             |
| Mistral Large 2 [Proprietary] | Mistral AI              | Proprietary             |

**Notes:**
- Mixtral 8x7b and 8x22b models have two sizes due to different total size and number of parameters used during inference.
- Some models were not utilized for certain experiments due to cost constraints, but the selected subset maintained model family diversity.

### 2.5 Prompt Templates

**LLM Prompts: Effective Templates for Social Media Post Analysis**

**Role of a Psychiatrist**:
- LLMs adopt role of psychiatrist through role-playing
- Encourages relevant and contextually appropriate responses
- Proven effective in eliciting responses from LLMs [71, 72]

**Prompt Templates**:
- Fixed structure with variable elements to accommodate diverse capabilities of LLMs
- Minimizes prompt-based bias while ensuring consistency across experiments

**Binary Disorder Detection Prompts**:
- BIN-1 to BIN-4: Assess impact of open-ended and structured prompts on various tested LLMs

**Disorder Severity Evaluation Prompts**:
- SEV-1 to SEV-4: Iteratively refined based on initial observations from binary task
- Added modifications to address verbosity in some expensive models

**Performance Comparison**:
- Improved performance for certain LLMs (Gemini 1.0 Pro, Llama 3) from P1 to P4
- Minor decreases for other models that did not show significant improvement

**Psychiatric Knowledge Assessment Prompt**:
- Simple prompt template (KNOW-1) with a multiple-choice question and no explanation

### 2.6 Evaluation Metrics

**Evaluation Metrics**
- Primary metric: Accuracy
  * Balanced sampling leads to Balanced Accuracy (BA) for most tasks
    **Assesses model's ability to identify both positive and negative instances**
    **Well-suited for classification tasks with class imbalance**
  * Traditional accuracy used when fair random sampling not feasible
- Disorder severity evaluation task: Two metrics
  * BA: Proportion of posts for which the LLM predicted exact severity level
  * Mean Absolute Error (MAE): Average magnitude of errors in predicted severity ratings
**Performance Metrics and Invalid Responses of Different Models**
| Model/Prompt | P1   | P2    | P3     | P4      | GPT-3.5-Turbo |
| ------------|-------|--------|--------|-----------|--------------|
| MAE: 0.713   |        |        |        |          |           |
| MAE: -0.001   |        |        |        |          |           |
| MAE: -0.029   |        |        |        |          |           |
| MAE: -0.066    |        |        |        |          |           |
| BA: 43.6%     |        |        |        |          |           |
| IR: 0         |        |        |        |          |           |
|-------------------|-------|--------|--------|-----------|--------------|
| Gemini 1.0     | MAE: 0.629 | MAE: -0.039 | MAE: -0.095 | MAE: -0.133   |           |
| BA: 42.7%     | +2.8%    | +7.6%    | +11.5%   |           |           |
| IR: 0         |        |        |        |          |           |
|-------------------|-------|--------|--------|-----------|--------------|
| Claude 3       | MAE: 0.97 | MAE: -0.041 | MAE: -0.069 | MAE: -0.092   |           |
| BA: 29.8%     | +2.3%    | +3.2%    | +4.5%    |           |           |
| IR: 70        |        |        |        |          |           |
|-------------------|-------|--------|--------|-----------|--------------|
| Mistral Medium | MAE: 0.608 | MAE: +0.001 | MAE: -0.051 | MAE: -0.052   |           |
| BA: 46.2%     | +2.2%    | +5.8%    | +5.7%    |           |           |
| IR: 38        |        |        |        |          |           |
|-------------------|-------|--------|--------|-----------|--------------|
| Gemma 7b       | MAE: 0.892 | MAE: +0.027 | MAE: +0.058 | MAE: +0.031   |           |
| BA: 35.5%     | -1%      | -1.9%    | -1.3%     |           |           |
| IR: 4         |        |        |        |          |           |
|-------------------|-------|--------|--------|-----------|--------------|
| Llama 3 8b      | MAE: 1.093 | MAE: +0.146 | MAE: -0.043 | MAE: -0.324   |           |
| BA: 32%        | -5.7%    | -1.3%     | +5%      |           |           |
| IR: 137       |        |        |        |          |           |
|-------------------|-------|--------|--------|-----------|--------------|
| Llama 3 70b     | MAE: 0.949 | MAE: -0.062 | MAE: -0.105 | MAE: -0.275   |           |
| BA: 29%        | +4.3%    | +6.0%     | +13.3%    |           |           |
| IR: 23        |        |        |        |          |           |

**Example of Analyzing MCQ Questions**
- Psychologist analyzes a given MCQ question and selects the most appropriate answer without explanation

### 2.7 Parsing Model Outputs

**Parsing Model Outputs**

**Challenges**:
- Interpreting and extracting meaningful information from LLM (Large Language Model) outputs was non-trivial
- Larger, more sophisticated models often deviated from explicit instructions regarding output format
  - Provided explanations or incorporated requested answer in longer response
- Adherence to instructions varied widely and did not necessarily correlate with model size

**Approach**:
- Developed custom parsers for each task
  - Binary Disorder Detection: searched for "yes" or "no", took first found as answer if only one present
    - If both or neither found, considered invalid response
  - Disorder Severity Evaluation: searched for numerical values in specified range
    - Accepted single valid number as answer
    - If multiple numbers, outside valid range, or no number found, considered invalid response
  - Psychiatric Knowledge Assessment: parsing multiple-choice answers more challenging
    - Employed iterative approach to refine parser's regular expressions
    - Revealed LLMs tend to follow limited set of patterns in responses, allowing for reliable parsing with few rules

**Interesting Findings**:
- Only eight rules required to parse majority of outputs across all LLMs.

### 2.8 Additional Considerations

**Additional Considerations for Language Models (LLMs)**
- **Modified prompts**: Incorporated a "repetition line" to improve adherence to instructions for some models that struggled with prompt engineering, like Mistral Medium and Claude 3 Opus.
- **Small, specialized experiments**: Presented separately in Section 4 for clarity and coherence, focusing on specific aspects of LLM performance or research questions.
- **Output token limit vs. free response**: Tested two approaches: setting a maximum output token limit of 2 tokens versus allowing models to respond freely without truncation. Imposed a response length limit of 2 tokens due to findings that longer responses were more likely to become invalid with certain models.

**Findings on Invalid Response Rates for Different LLMs**:
- **GPT Family**: Strong performance, average invalid response rate was 0.45%; GPT-4 showed impressive improvement at 0.02%, but GPT-4 Turbo had an increased rate of 1.64%; Subsequent versions, GPT-4o and GPT-4o mini, returned to low rates of 0.22% and 0%.
- **Llama Family**: Llama 3 represented a marginal improvement over Llama 2 (0.43% to 6.37%), but recent releases like Llama 3.1 70b and 405b showed significantly worse performance; Llama 3.1 70b had a 0.42% worse performance than Llama 3 70b, while other models in the family displayed a range of invalid response rates.
- **Mistral Family**: Steady improvement in following instructions, starting with Mistral 7b at 12.29% and decreasing to nearly 0%; However, Mistral Large unexpectedly had a high invalid response rate of 19.68%, which was later corrected in Mistral Large 2.
- **Gemma Family**: Stable pattern in invalid response rates, with most models performing consistently across iterations; The notable exception was Gemma 2b, which recorded the highest invalid rate at 33.53%; However, larger models like Gemma 1.0 Pro, Gemma 2 9b, and Gemma 27b showed significantly lower rates.
- **Phi, Qwen, Claude Families**: Varied performances within each family; Phi models showed a range of invalid response rates from 0.07% to 27%, with Phi-3-medium being the best; The Qwen 2 72b model had a moderate invalid response rate of 1.46%; Claude family displayed significant variability, ranging from 0.8% to 20.49%. Surprisingly, Claude 3 Opus showed worse performance compared to the cheaper Clause 3 Haiku; Both being significantly worse than Claude 3 Sonnet.

## 3 Results & Discussion

**Performance Analysis of Various LLMs**
- Presenting results of experiments on three tasks:
  - ZS binary disorder detection task (Section 3.3)
  - ZS/FS disorder severity evaluation task (Section 3.4)
  - ZS psychiatric knowledge assessment task (Section 3.5)

**Performance Variability Analysis:**
- Examining performance variability between runs of different LLMs using consistent parameters
- Establishing reliability and reproducibility foundation for assessing validity of improvements between prompts

**ZS Binary Disorder Detection Task Results (Section 3.3):**
- GPT-4, GPT-4o, Mistral NeMo, and Llama 2 70b consistently outperformed other models
- Prompt choice significantly influenced model performance
  - Structured vs open-ended prompts: Gemma 7b achieved higher accuracy (76.70%) than previous best (Mistral NeMo) of 74.50%

**ZS/FS Disorder Severity Evaluation Task Results (Section 3.4):**
- Most models benefited from few-shot learning, with varying degrees of improvement across models and datasets
- Careful consideration required for both model architecture and data characteristics

**ZS Psychiatric Knowledge Assessment Task Results (Section 3.5):**
- More recent models generally performed better compared to older, much larger ones (e.g., Llama 3.7b and Phi-3-medium vs GPT-4, GPT-4-Turbo)
- With models around the same release time period, model size seemed to be the deciding factor (excluding Llama 3.1 due to sensitive nature)

### 3.2 Performance Variability Experiment

**Performance Variability Experiment**

**Methodology**:
- Evaluate inherent performance variability of each Language Model (LLM) on same task
- Repeatedly evaluate models on the same dataset 5 times using standard parameters (e.g., temperature 0)
- Approximate natural performance fluctuations to judge meaningful improvement or random fluctuation due to model's inherent variability

**Data Set**:
- Conducted on 1000 fairly sampled instances from DEPTWEET dataset

**Results**:

| Model          | Standard Deviation (percent) |
|----------------|----------------------------|
| Gemma          | **0.5%**                   |
| Gemma (27b)    | **1.0%**                   |
| GPT-3.5-Turbo  | **0.4%**                   |
| Llama (2)       | **0.2%**                   |
| Llama (3)       | **0.1%**                   |
| Mixtral (8x22b)| **1.6%**                   |
| Phi-3-small    | **0.05%**                  |
| Phi-3-medium   | **0.16%**                  |
| Qwen          | **0.4%**                   |

**Observations**:
- Most models exhibited performance fluctuations less than **0.5%**
- Mixtral (8x22b) showed a **1.6%** deviation
- Models not listed had a standard deviation of **0%**, indicating no performance variability across 5 runs

**Conclusion**:
- Observed changes in model performance larger than observed fluctuations are likely due to modifications (e.g., prompt) rather than inherent model variability.

### 3.3 Binary Disorder Classification (Task 1)

**Binary Disorder Classification (Task 1)**

**ZS binary disorder detection task**:
- Evaluated various LLMs' performance in identifying presence or absence of:
  - Depression
  - Suicide risk
  - Stress
- Used the **BIN-1 prompt template** to begin assessment
- Analyzed impact of modifications introduced in prompts:
  - **BIN-2**
  - **BIN-3**
  - **BIN-4**

#### 3.3.1 Results of Prompt BIN-1

**Model Performance Analysis**

**General Trend**:
- OpenAI's GPT-4, GPT-4o, and GPT-4-Turbo models outperform other tested models on most datasets
- Llama 2 70b performs best in DepSeverity dataset with 73.58% accuracy
- Mistral NeMo leads in Dreaddit Test and RED SAM with 74.50% and 66.00% accuracy, respectively

**Model Comparisons**:
- GPT-4 achieves highest accuracy of approximately 85% on DEPTWEET and SAD datasets
- GPT-4-Turbo lags slightly behind GPT-4 but outperforms in two instances (63.7% vs. 62.1% in RED SAM, 70.8% vs. 69.3% in SDCNL)
- GPT-4o exhibits significant fluctuations, sometimes exceeding all other variants (e.g., 71.2% in SDCNL) but also deviating from GPT-4 by up to 9.8% in SDCNL
- GPT-3.5-Turbo follows closely behind GPT-4 with a maximum deviation of approximately 5%
- The newest and most cost-efficient model, GPT-4o mini, generally performs worse compared to GPT-3.5-Turbo
- Llama 2 70b outperforms the newer Llama 3 models in DepSeverity but is surpassed by the Llama 3 family for consistency
- Llama 3.1 8b and 70b generally perform better than earlier versions, with notable exceptions in SAD and DEPTWEET datasets
- The Llama 3.1 405b model performs worse than expected across all six datasets
- Mistral 7b shows moderate performance but outperforms Llama 2 13b and is surpassed by Llama 3 8b
- The newer Mistral NeMo model significantly outperforms other models, especially on Dreaddit Test dataset with 74.5% accuracy

**Other Models**:
- Claude family demonstrates varied results, but Claude 3.5 Sonnet is the best-performing variant
- Gemini 1.0 Pro generally trails GPT-3.5-Turbo but slightly outperforms in DEPTWEET dataset
- Gemma 2 models show inconsistency between their 9b and 27b variants, with 9b model often outperforming its larger sibling
- Phi-3-mini model consistently outperforms the larger Phi-3-small model in most datasets

**Conclusion**:
- OpenAI's GPT-4 stands out for its exceptional performance across multiple datasets
- Llama 3 70b leads in sub-70 billion parameter category, followed closely by Llama 3.1 70b
- Phi-3-medium and Gemma 2 9b perform admirably despite their relatively small sizes
- Mistral NeMo emerges as the top model in sub-14 billion parameter category.

#### 3.3.2 Results of Prompts BIN-2/3/4

**Effects of Prompts BIN-2/3/4 on Model Performance**

**Changes from Prompt BIN-1 to BIN-2**:
- Changing main task from checking for disorder to checking symptoms
- Creates less strict boundary for true label
- Models like Gemini 1.0 Pro, Mistral Medium, etc., showed improvements on Dreaddit and DepSeverity datasets, especially for models with initially lower performance

**Transitions from BIN-1 to BIN-3/4**:
- Fewer models exhibited improved performance
- Notable declines in performance for some models, like GPT-3.5-Turbo, Claude 3 Haiku, etc.
- Preference for BIN-2 over BIN-3/4 due to clearer approach

**Models Showing Significant Improvements**:
- Gemini 1.0 Pro: +8% (Dreaddit), +4% (DepSeverity) with BIN-4
- Gemma 7b: 22.58% increase on Dreaddit, 11.88% (DepSeverity) with BIN-3/4
- Llama 3.1: 9.41% (Dreaddit), 13.80% (DepSeverity) with BIN-4
- Mistral Medium, Mixtral 8x7b/22b, Mistral Large 2, Gemma 7b, and Phi-3-medium displayed fluctuations but significant improvements with some prompts

**Overall Best Performing Models**:
- **Gemma 7b**: Top performer on Dreaddit dataset, 76.70% accuracy (BIN-4)
- Other notable performances: Gemini 1.0 Pro, Mixtral 8x7b/22b, and Mistral Medium

### 3.4 Disorder Severity Evaluation (task 2)

**Disorder Severity Evaluation (Task 2)**

#### Performance Comparison of Proprietary LLM on SEV-4 Task
- GPT family models exhibit strong performance across multiple tasks:
  - **GPT-4-Turbo**: Achieves the highest BA of 39.6% in DepSeverity and an outstanding 59.7% in DEPTWEET, making it the top performer for these datasets. Its MAE scores are notably low, with a minimum of 0.455 in DEPTWEET.
  - **GPT-4**: Performs admirably, with a notable BA of 40.9% in RED SAM Test and 29.0% in SAD. Consistent MAE scores contribute to an average BA of 40.48% and MAE of 1.141.
- **GPT-4o** demonstrates competitive results with a BA of 35.6% in DepSeverity and 50.1% in DEPTWEET, complemented by a low MAE of 0.825.
- **GPT-3.5-Turbo** delivers moderate performance, peaking at 43.2% BA in DEPTWEET and maintaining consistent MAE scores.
- The **GPT-4o mini**, despite its more compact architecture, falls slightly behind the larger GPT-4o model in terms of performance.

**Merged Results for Prompt BIN-2/BIN-3/BIN-4 on Task 1**:
- **Claude 3 Haiku** shows generally poor results with low average BA and high MAE, positioning it among the weaker models in this analysis.
- **Claude 3 Sonnet** stands out, particularly in the SAD dataset, where it achieves the highest BA of 29.7% and the lowest MAE of 1.404. It also performs strongly in DEPTWEET with a BA of 45.3% and an MAE of 0.683, resulting in an overall average BA of 35.73% and MAE of 0.975.
- **Claude 3 Opus** matches the highest BA in SAD (29.7%) and excels with a BA of 50.6% in DEPTWEET, along with a low MAE of 0.547.

#### Performance Comparison of Open Source LLM on SEV-4 Task
**Gemini**:
- Gemini 1.0 Pro achieved significant scores, including a BA of 54.2% in DEPTWEET and an MAE of 0.729 in the RED SAM Test
- Average scores: 36.63% BA and 1.122 MAE

**Gemma**:
- Gemma models, particularly 2 9b, maintained consistent and competitive performance
- Average scores: 33.80% BA and 1.243 MAE

**Mistral**:
- Mistral Medium stood out with a BA of 39.0% in DepSeverity and the best MAE of 0.779
- Mistral Large and Mistral 7b delivered moderate performances, with BA scores of 31.5% and 31.9% in DepSeverity, respectively
- **Mistral NeMo** impressed with the best MAE of 0.639 in the RED SAM Test and a commendable MAE of 1.738 in SAD, resulting in an overall MAE of 1.006
- However, Mistral Large 2 underperformed compared to its predecessors

**Llama**:
- Llama 3 70b emerged as the best among them, achieving a BA of 33.4% in DepSeverity and showing competitive scores across other datasets
- In contrast, Llama 2 70b underperformed significantly, with a BA of just 12.7% in DepSeverity and high MAE scores
- The newer Llama 3.1 family failed to surpass their Llama 3 counterparts

**Phi-3**:
- Phi-3-mini demonstrated moderate capabilities with a BA of 31.6% in DepSeverity and reasonable MAE scores
- Phi-3-small exhibited lower performance, with a BA of 25.5% in the same task
- Phi-3-medium performed slightly better, achieving a BA of 32.1% in DepSeverity and maintaining competitive scores across other datasets

**Qwen 2 72b**:
- Delivered moderate overall performance, comparable to models like Llama 3 70b, Gemma 2 9b, or Phi-3-medium

**Table of Results**:
- Provided for Prompt SEV-4 on Task 2 (ZS) for various models and datasets.

#### 3.4.2 Results of prompt

**FS Severity Evaluation Task Results for Various Models**

**Table 10 Summary**:
- Most models improved on various datasets with FS prompt
- Some models experienced minor decreases
- Even small improvements in all datasets except SAD could result in negative final average change due to significant impact of SAD's 10 severity levels

**Claude 3.5 Sonnet**:
- Significant improvements: overall +6.50% BA, -0.248 MAE
- Mixed results: DEPTWEET (+5.30% BA, -0.058 MAE), SAD (-14.4% BA, +1.397 MAE)
- Overall negative change due to decrease in SAD

**Claude 3 Opus**:
- Minimal improvements
- Negative overall performance due to decreases in RED SAM Test and SAD
- Average decrease of -2.70% BA, increase of +0.096 MAE

**Gemma Models**:
- Substantial positive changes: Gemma 2 27b (+6.05% BA, -0.191 MAE), Gemma 2 9b (+2.175% BA, -0.131 MAE)
- Consistent improvements across datasets for Phi-3-mini and Phi-3-medium
- Notable changes: GPT-4o excels in MAE across all datasets except RED SAM; Gemini 1.0 Pro dominates RED SAM, Mistral 7b improved performance for SAD dataset

**SAD's Impact**:
- With inclusion of three additional example posts, overall token count per evaluation is multiplied by 3-4
- GPT-4-Turbo now excels in MAE across all datasets except RED SAM
- SAD's significant impact can disproportionately influence final average changes.

### 3.5 Psychiatric Knowledge Assessment (task 3)

**Psychiatric Knowledge Assessment (Task 3)**

**Models' Performance**:
- Llama 3.1 405b tops the performance chart with an accuracy of 91.2%
- Llama 3.1 70b follows closely behind at 89.50% accuracy
- Smaller Llama 3.1 8b lags behind with an accuracy of 68.0% but outperforms other models in its size category
- Older Llama 3 models perform slightly worse compared to more recent models

**Comparison with Other Models**:
- GPT-4o, GPT-4-Turbo, and GPT-4 follow behind the Llama 3.1 models but are better trained and have a better overall understanding of psychiatric knowledge
- Phi-3-medium model is impressive for its small size (14 billion parameters) compared to larger models
- Claude 3 Opus slightly surpasses Mistral Large, with an accuracy of 81.8% vs. 78.9%
- Gemma 2 27b follows closely with an accuracy of 78.6%, nearly matching Mistral Large despite its much smaller size

**Trends Observed**:
- Model size matters, but more recent models perform significantly better than their older counterparts, even with massive size differences
- Within the same release period, model size remains a critical factor when training data quality is comparable
- Newer models are often trained on larger and more diverse datasets, enhancing their foundational knowledge in psychiatry

## 4 Additional Investigations

**Experiment 1: Repetition Line Effect**
- **Objective**: Evaluate impact of repetition line in binary task
- **Methodology**: Appended line to end of BIN-1 prompt, analyzed performance of various models (Claude 3 Haiku - Mistral Large)
- **Findings**: Improvements ranging from 1% to 14%; larger improvements for Mistral family models
- **Interpretation**: Some models prefer reiterating instructions or important information in prompts

**Experiment 2: Knowledge Reinforcement**
- **Objective**: Investigate impact of providing models with their own generated information on performance
- **Methodology**: LLMs listed symptoms for depression, appended to modified BIN-1 prompt (Knowledge Reinforcement Prompt) with repetition line and symptom list
- **Findings**: Mixed results; some models showed increase in accuracy (GPT-3.5-Turbo, Claude 3 Sonnet), others experienced decrease (most smaller LLMs); unclear reason for negative effect on smaller models

## 5 Challenges, Conclusions and Future Directions

**Challenges Faced in Evaluating LLMs for Mental Health Applications**

**Dataset Limitations**:
- Ideal evaluation requires clinical reports containing patient summaries and diagnoses, but these are difficult or impossible to access due to confidentiality concerns
- Reliance on publicly available social media datasets:
  - Potential noise from context-based labels (e.g., subreddit of origin)
  - Human supervision used to mitigate this issue, but psychiatric evaluation remains subjective and can introduce variability

**LLM Limitations**:
- Cost associated with models like GPT-4 and Claude Opus limited experiments to samples from test sets
- Context limit in some models (e.g., Phi-2) made it difficult to handle datasets with very long posts
- API filtering policies, such as Gemini 1.0 Pro's restriction on sensitive topics, limit the model's applicability in mental health contexts
- Ethical guards implemented in some models (e.g., Llama 3.1) have been problematic and hinder performance compared to older counterparts
- Some models exhibit stubborn behavior, resisting providing explanations despite clear instructions
- Sensitivity of models to prompt changes can lead to excessive sensitivity

**Future Directions**:
- Application of CoT prompting to enhance interpretability and decision-making capabilities of LLMs in psychiatric contexts
- Use of Retrieval-Augmented Generation (RAG) techniques to provide additional context and grounding to model responses, enhancing diagnostic accuracy and relevance
- Fine-tuning LLMs on specific psychiatric datasets to improve performance and generalizability across different mental health conditions
- Evaluating LLMs with datasets in other languages or modalities (e.g., audio, images, videos) to enhance diagnostic accuracy and offer a richer understanding of mental health conditions

**Conclusions**:
- LLMs hold significant promise in assisting with mental health tasks
- GPT-4 and Llama 3 exhibited superior performance in binary disorder detection, achieving accuracies up to 85% on certain datasets
- Challenges, such as social media dataset limitations, high costs of large models, and ethical restrictions, must be addressed to effectively deploy LLMs in mental health applications.

